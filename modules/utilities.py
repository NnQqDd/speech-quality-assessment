import importlib
import wandb


def load_class(full_name: str): # or use eval to convert to class
    module_name, class_name = full_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def print_num_params(model, show_per_module=False):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,} ({total/1e6:.3f} M)")
    print(f"Trainable params: {trainable:,} ({trainable/1e6:.3f} M)")
    if show_per_module:
        print("\nPer-module parameter counts:")
        for name, module in model.named_children():
            p = sum(p.numel() for p in module.parameters())
            t = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:20s} | total: {p:12,} | trainable: {t:12,}")



def remove_empty_runs(entity: str, project: str, least: int = 1, skip_active: bool = True) -> list:
    """
    Delete runs in `entity/project` that have fewer than `least` history rows and are not active.
    Returns a list of deleted run IDs.
    """
    api = wandb.Api()
    deleted_runs = []
    active_states = {"running", "starting", "created", "queued", "resuming", "scheduled"}

    for run in api.runs(f"{entity}/{project}"):
        try:
            state = getattr(run, "state", None)
            if state is None:
                info = getattr(run, "info", None)
                if isinstance(info, dict):
                    state = info.get("state")

            if skip_active and state and str(state).lower() in active_states:
                continue

            rows = 0
            try:
                hist = run.history()
                rows = len(hist) if hist is not None else 0
            except Exception:
                try:
                    summary = dict(run.summary)
                    rows = sum(1 for v in summary.values() if v is not None)
                except Exception:
                    rows = 0

            if rows < least:
                try:
                    run.delete()
                    deleted_runs.append(run.id)  # Collect run ID
                except Exception:
                    pass
        except Exception:
            pass

    return deleted_runs


