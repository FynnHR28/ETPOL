import pandas as pd
import torch
from torch.utils.data import Dataset
import re
from string import printable

# helpful function to apply clean any clean up model input data consistently 
def clean_text(text):
    # normalize to lowercase -> more efficient tokenization
    text = text.lower()
    # remove anything that is not alphanumeric or punctuation
    re.sub(r'[^\x20-\x7E]', '', text)
    text = re.sub(r'[^a-z0-9\s.,:;!?()\'"-]', '', text)
    text = text.replace('amp;', '&')
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\n', ' ', text)
    # remove leading/trailing whitespace
    text = text.strip()
    text = ''.join(char for char in text if char in printable)
    
    return text


# following https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for creating a custom dataset to help with tokenization
class PostsDataset(Dataset):
    def __init__(self, df, tokenizer, context_length):
        self.df = df
        self.tokenizer = tokenizer
        self.context_length = context_length
        mapping_dict = {'left': 1, 'right': 0}
        # convert string labels to numerical labels based on mapping dict
        self.df['affiliation'] = self.df['affiliation'].map(mapping_dict)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # extract the idx row and tokenize it, returning torch tensors
        content = self.df.iloc[idx]['content']
        label = self.df.iloc[idx]['affiliation']
        tokenized = self.tokenizer(
            content,
            max_length=self.context_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # returning in the form of a dictionary helps with ease of training
        return {
            # squeeze to get rid of the default batch dimension added by tokenizer
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    