import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from models import ETPOL
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from PostsDataset import PostsDataset
from test_utils import train, save_results



if __name__ in '__main__':
    
    """Model & training hyper parameters"""
    context_length = 512
    d_model = 512
    num_heads = 8
    num_hidden_layers = 2
    d_hidden = 2048
    num_encoders = 8
    num_epochs = 2

    lr = 2e-4
    batch_size = 16
    
    model = ETPOL(
        vocab_size=20000,
        context_length=context_length,
        d_model=d_model,
        d_hidden=d_hidden,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
        num_encoders=num_encoders   
    )

    model_hyperparams = {
        'context_length': context_length,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_hidden_layers': num_hidden_layers,
        'd_hidden': d_hidden,
        'num_encoders': num_encoders,   
    }
    
    model_hyperparams_str = '\n'.join(f'{k}: {v}' for k, v in model_hyperparams.items())


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # good for binary classification
    loss_fn = nn.CrossEntropyLoss()

    print('loading tokenizer')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(Path('wp_tokenizer'), model_max_lenth=context_length)

    print('loading df..')
    df=pd.read_csv(Path('data/all_posts.csv'))
    df = df[:32]

    print('splitting into train and test sets')
    train_df, temp = train_test_split(df, test_size=0.3, stratify=df['affiliation'])
    val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp['affiliation'])

    print('creating datasets..')
    train_dataset = PostsDataset(df=train_df, tokenizer=tokenizer, context_length=context_length)
    val_dataset = PostsDataset(df=val_df, tokenizer=tokenizer, context_length=context_length)
    test_dataset = PostsDataset(test_df, tokenizer=tokenizer, context_length=context_length)

    print('creating dataloaders')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        loss_fn=loss_fn
    )
    
    save_results(results, model_hyperparams_str, 'test_1')
                    
                    
