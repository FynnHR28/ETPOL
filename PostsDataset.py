import pandas as pd
import torch
from torch.utils.data import Dataset

# following https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class PostsDataset(Dataset):
    def __init__(self, df, tokenizer, context_length):
        self.df = df
        self.tokenizer = tokenizer
        self.context_length = context_length
        mapping_dict = {'left': 1, 'right': 0}
        self.df['affiliation'] = self.df['affiliation'].map(mapping_dict)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        content = self.df.iloc[idx]['content']
        label = self.df.iloc[idx]['affiliation']
        tokenized = self.tokenizer(
            content,
            max_length=self.context_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            # squeeze to get rid of the default batch dimension added by tokenizer
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    