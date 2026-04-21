import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


if __name__ == "__main__":
    METADATA_DIRECTORY = "/home/duyn/ActableDuy/sqa/dataset/metadatas"
    AUDIO_DIRECTORY = "/home/duyn/ActableDuy/sqa/dataset/audios"
    AUDIO_TYPE = ".wav"
    
    metadata_txts = [os.path.join(METADATA_DIRECTORY, d) for d in os.listdir(METADATA_DIRECTORY)]
    metadatas = []
    for txt in tqdm(metadata_txts):
        with open(os.path.join(METADATA_DIRECTORY, txt), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            metadatas.append([os.path.join(AUDIO_DIRECTORY, lines[-1] + AUDIO_TYPE), str(lines[1]), float(lines[-2])])

    metadata_df = pd.DataFrame(metadatas, columns=["filepath", "speaker_id", "label"])
    print(metadata_df.nunique())

    metadata_df['label'] = metadata_df['label'] / metadata_df['label'].max()
    speakers = metadata_df["speaker_id"].unique()

    # 80% train, 20% temp
    spk_train, spk_temp = train_test_split(
        speakers,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # split remaining 20% → 10% val, 10% test
    spk_val, spk_test = train_test_split(
        spk_temp,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    # assign splits
    df_train = metadata_df[metadata_df["speaker_id"].isin(spk_train)]
    df_val   = metadata_df[metadata_df["speaker_id"].isin(spk_val)]
    df_test  = metadata_df[metadata_df["speaker_id"].isin(spk_test)]

    # combine
    metadata_df = pd.concat([
        df_train.assign(split="train"),
        df_val.assign(split="valid"),
        df_test.assign(split="test")
    ]).reset_index(drop=True)

    # save
    metadata_df.to_csv("metadatas/metadata.csv", index=False)