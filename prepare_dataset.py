import os
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    VOICE_CONVERSION_PATH = "/home/duyn/ActableDuy/voice-synthesis/voice-conversion-audios"
    VOICE_CLONE_PATH = "/home/duyn/ActableDuy/voice-synthesis/voice-clone-audios"
    metadatas = []
    code_to_label = {}
    for path in [VOICE_CONVERSION_PATH, VOICE_CLONE_PATH]:
        for model_code in os.listdir(path):
            for audio_name in os.listdir(os.path.join(path, model_code)):
                if model_code not in code_to_label:
                    code_to_label[model_code] = len(code_to_label)
                metadatas.append((os.path.join(path, model_code, audio_name), model_code, code_to_label[model_code]))

    metadata_df = pd.DataFrame(metadatas, columns=["filepath", "model", "label"])
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

    metadata_df.to_csv("metadata.csv", index=False)