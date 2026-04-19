import os
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    METADATA_DIRECTORY = "/home/duyn/ActableDuy/sqa/dataset/metadatas"
    AUDIO_DIRECTORY = "/home/duyn/ActableDuy/sqa/dataset/audios"
    AUDIO_TYPE = ".wav"
    
    metadata_txts = [os.path.join(METADATA_DIRECTORY, d) for d in os.listdir(METADATA_DIRECTORY)]
    metadatas = []
    for txt in metadata_txts:
        with open(os.path.join(METADATA_DIRECTORY, txt), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            metadatas.append([os.path.join(AUDIO_DIRECTORY, lines[-1] + AUDIO_TYPE), float(lines[-2])])

    metadata_df = pd.DataFrame(metadatas, columns=["filepath", "label"])
    metadata_df['label'] = metadata_df['label'] / metadata_df['label'].max()

    train_dfs = []
    val_dfs = []
    for label, group in metadata_df.groupby("label"):
        train, val = train_test_split(
            group,
            test_size=0.1,
            random_state=42,
            shuffle=True
        )
        train_dfs.append(train)
        val_dfs.append(val)

    df_train = pd.concat(train_dfs).reset_index(drop=True)
    df_val   = pd.concat(val_dfs).reset_index(drop=True)
    
    test_dfs = []
    val_dfs = []
    for label, group in df_val.groupby("label"):
        val, test = train_test_split(
            group,
            test_size=0.5,
            random_state=42,
            shuffle=True
        )
        val_dfs.append(val)
        test_dfs.append(test)
    df_val = pd.concat(val_dfs).reset_index(drop=True)
    df_test = pd.concat(test_dfs).reset_index(drop=True)

    metadata_df = pd.concat([
        df_train.assign(split="train"),
        df_val.assign(split="valid"),
        df_test.assign(split="test")
    ])

    metadata_df.to_csv("metadatas/metadata.csv", index=False)