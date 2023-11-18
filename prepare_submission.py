import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda"

import torch
import pandas as pd
from tqdm import tqdm
from train import DisinformationBERT

model = DisinformationBERT.load_from_checkpoint("/mnt/vol_d/bh/checkpoints/epoch=14-val_loss=0.94.ckpt")

print("created model")

df = pd.read_csv("test.csv")
df = df.drop(columns=["ChannelId", "ChannelName", "Date", "EditDate"])

train_encodings = model.tokenizer(list(df["Content"]), truncation=True, padding=True, max_length=512)

print("tokenized test subset")

predictions = []
for idx in tqdm(range(df.shape[0])):
    input_ids = torch.tensor(train_encodings["input_ids"][idx], device=device).unsqueeze(0)
    attn_mask = torch.tensor(train_encodings["attention_mask"][idx], device=device).unsqueeze(0)
    with torch.no_grad():
        pred = model(
            torch.tensor(train_encodings["input_ids"][idx], device=device).unsqueeze(0),
            attention_mask=torch.tensor(train_encodings["attention_mask"][idx], device=device).unsqueeze(0)
        )

    pred = torch.argmax(pred, dim=1).item()

    predictions.append(pred)

df = df.drop(columns=["Content"])
df["Suspicious_Level"] = predictions
df.to_csv("first_submission.csv", index=False)
