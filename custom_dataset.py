import torch
from torch.utils.data import Dataset
import pandas as pd


class CustomDataset(Dataset):   
    def __init__(self, file_path, tokenizer, seqlen, num_samples=None, mask_padding_with_zero=True):

        # 读取原始数据
        df = pd.read_csv(file_path,index_col=None)

        self.texts = df["text"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = seqlen
        self.num_samples = num_samples

        all_labels = list(set([e for e in self.labels]))
        self.label_to_idx = {e: i for i, e in enumerate(sorted(all_labels))}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    
    def __len__(self):
        if self.num_samples and len(self.texts) > self.num_samples:
            return self.num_samples
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        tokenized = self.tokenizer.encode_plus(
            text=text,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True
        )
        return (tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), label)