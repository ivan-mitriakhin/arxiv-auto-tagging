import pandas as pd
import numpy as np
import torch

class AbstractsDataset():
    
    def __init__(self, data, tokenizer, max_seq_len):
        self.data = data
        self.titles = data['titles'].tolist()
        self.texts = data['abstracts'].tolist()
        self.targets = None
        if data.shape[1] > 3:
            self.targets = data.drop(['ids', 'titles', 'abstracts'], axis=1)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        title = str(self.titles[index])
        title = ' '.join(title.split())

        text = str(self.texts[index])
        text = ' '.join(text.split())

        encoding = self.tokenizer.encode_plus(
            title,
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_token_type_ids=True
        )

        ids = encoding['input_ids']
        mask = encoding['attention_mask']
        token_type_ids = encoding['token_type_ids']

        if self.targets is None:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
            }
        else:
            targets = self.targets.iloc[index].tolist()
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(targets, dtype=torch.long)
            }