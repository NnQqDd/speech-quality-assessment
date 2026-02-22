import dotenv; dotenv.load_dotenv()
import random

import warnings
warnings.filterwarnings("ignore")

import os
import re
import argparse
import datetime
import yaml
from setproctitle import setproctitle; setproctitle("SQA Training")
from loguru import logger
# logger.remove(); logger.add(sys.stdout)
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from modules.utilities import *
from modules.dataset import prepare_dataloaders


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cf", type=str, help="Config path")
    parser.add_argument("--ckpt", "-ck", type=str, help="Checkpoint path")
    args = parser.parse_args()

    BASE_PATH = os.path.abspath(os.path.dirname(__file__))
    fill_path = lambda x: os.path.join(BASE_PATH, x)

    config_path = fill_path('config.yaml') if args.config is None else args.config 
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    wandb.login(key=config['wandb']['api_key'])
    entity = config['wandb']['entity']
    project = config['wandb']['project']

    try:
        deleted_ids  = remove_empty_runs(entity, project, least=1)
        logger.info(f'{len(deleted_ids)} empty run(s) deleted.')
    except Exception as e:
        logger.info(e)

    run = wandb.init(
        dir=BASE_PATH, 
        project=project,
        config=config,
        entity=entity,
        name=datetime.datetime.now().strftime("%m.%d.%H.%M.%S"),
        # mode="offline"
    )
    run_id = run.id
    os.makedirs(fill_path(f'weights/{run_id}'), exist_ok=True)
    with open(fill_path(f'weights/{run_id}/config.yaml'), 'w') as f:  
        yaml.dump(config, f, default_flow_style=False)  
    
    if os.environ.get("WANDB_MODE", "") == "disabled":
        logger.info(f"Run ID = {run_id}.")

    logger.info("Preparing datasets and dataloaders...")
    dataloaders, num_classes = prepare_dataloaders(config['datasets'], config['dataloaders'])
    logger.info(f"Found {num_classes} class(es).")

    device = torch.device('cuda')
    logger.info(f'Current device has {torch.cuda.mem_get_info(device)[0]/(1024 ** 3):.2f} GB free memory.')

    ModelClass = load_class(config['model']['name'])
    if 'args' in config['model'] and config['model'] is not None:
        config['model']['args']['num_classes'] = num_classes
        model = ModelClass(**config['model']['args'])
    else:
        model = ModelClass(num_classes=num_classes)
    assert isinstance(model, torch.nn.Module)
    print_num_params(model)
    model.to(device)

    OptimizerClass = load_class(config['optimizer']['name'])
    opt_dicts = []
    if 'groups' in config['optimizer']:
        for group in config['optimizer']['groups']:
            group['params'] = getattr(model, group['name']).parameters()
            del group['name']
            opt_dicts.append(group)
        optimizer = OptimizerClass(opt_dicts)
    else:
        optimizer = OptimizerClass(model.parameters(), **config['optimizer']['args'])
    assert isinstance(optimizer, torch.optim.Optimizer)

    SchedulerClass = load_class(config['scheduler']['name'])
    scheduler = SchedulerClass(optimizer, **config['scheduler']['args'])

    CriterionClass = load_class(config['criterion']['name'])
    criterion = CriterionClass(**config['criterion']['args'])
    assert isinstance(criterion, torch.nn.Module)
    
    curr_epoch = 0
    if args.ckpt is not None and os.path.exists(args.ckpt):
        import re
        pattern = re.compile(r'epoch_(\d+)\.pth$')
        matches = pattern.findall(args.ckpt)   # returns list of all matches
        if matches:
            curr_epoch = int(matches[-1])      # last match
        logger.info(f'Recovering from {args.ckpt}, epoch {curr_epoch}...')
        state_dicts = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state_dicts['model_state_dict'])
        optimizer.load_state_dict(state_dicts['optimizer_state_dict'])
        scheduler.load_state_dict(state_dicts['scheduler_state_dict'])
    
    if config['trainer']['deterministic'] == True:
        set_seed(42)
    best_valid_loss = float('inf')
    # criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, config['trainer']['epochs'] + 1):
        if epoch <= curr_epoch:
            continue
        model.train()
        train_loss = 0.0
        iter = enumerate(dataloaders['train'])
        iter = tqdm(iter, total=len(dataloaders['train']), desc=f"Epoch {epoch} - Train")
        for idx, (waveforms, labels) in iter:
            # print(waveforms)
            optimizer.zero_grad()
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            preds = model(waveforms)
            preds = preds.squeeze(-1)
            loss = criterion(preds, labels)
            # print(preds.argmax(dim=1))
            # print(labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()                
            iter.set_postfix(loss=train_loss/(idx + 1))
        avg_train_loss = train_loss / (idx + 1)
        
        model.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            iter = enumerate(dataloaders['valid'])
            iter = tqdm(iter, total=len(dataloaders['valid']), desc=f"Epoch {epoch} - Valid")
            for idx, (waveforms, labels) in iter:
                waveforms = waveforms.to(device) # (batch_size, wave_length)
                labels = labels.to(device)       # (batch_size,)
                preds = model(waveforms)
                preds = preds.squeeze(-1)
                loss = criterion(preds, labels)
                valid_loss += loss.item()
                pred_classes = preds.argmax(dim=1)   # (B,)
                valid_acc += (pred_classes == labels).float().mean().item()
                iter.set_postfix(loss=valid_loss/(idx + 1), acc=valid_acc*100/(idx + 1))
        avg_valid_loss = valid_loss / (idx + 1)
        avg_valid_acc = valid_acc / (idx + 1)

        wandb.log({
            "Metrics/Valid Accuracy": avg_valid_acc,
        }, step=epoch)
        wandb.log({
            "Metrics/Train Loss": avg_train_loss,
            "Metrics/Valid Loss": avg_valid_loss,
        }, step=epoch)
        for idx, pg in enumerate(optimizer.param_groups):
            wandb.log({f"Others/LR {idx}": pg["lr"]}, step=epoch)

        if config['scheduler']['step'] is not None:
            scheduler.step(avg_valid_loss)
        else:
            scheduler.step()

        if avg_valid_loss < best_valid_loss:
            logger.info(f'Epoch {epoch}: best {avg_valid_loss:.4f} (previous: {best_valid_loss:.4f}).')
            logger.info(f'Epoch {epoch}: accuracy {avg_valid_acc:.4f}.')
            best_valid_loss = avg_valid_loss
            state_dict = model.state_dict()
            torch.save(state_dict, fill_path(f'weights/{run_id}/best.pth'))
            logger.info(f'Saved model.')
        
        if epoch % config['trainer']['save_frequency'] == 0:
            model_state_dict = model.state_dict()
            ckpt = {
                'model_state_dict': model_state_dict,      
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(ckpt, fill_path(f'weights/{run_id}/epoch_{epoch}.pth'))
            logger.info(f'Saved checkpoint in weights/{run_id}')   

    wandb.finish()
