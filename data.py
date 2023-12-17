import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def create_classification_dataset(path: str, tokenizer, max_len=512, test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    labels = df['Suspicious_Level'].astype(int)
    texts = df['Content'].astype(str)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )

    train_tokens = tokenizer(list(train_texts), truncation=True, padding=True, max_length=max_len)
    val_tokens = tokenizer(list(val_texts), truncation=True, padding=True, max_length=max_len)

    train_dataset = CustomDataset(train_tokens, torch.tensor(train_labels.values, dtype=torch.long) - 1)
    val_dataset = CustomDataset(val_tokens, torch.tensor(val_labels.values, dtype=torch.long) - 1)

    return train_dataset, val_dataset
