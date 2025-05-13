import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from models import ETPOL
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split  
from PostsDataset import PostsDataset
from test_utils import train, save_results, update_results, evaluate_model
import numpy as np




if __name__ in '__main__':
    
    # setting seed to ensure reproducability, without this its like shooting in the dark with no eyes ears or nose
    seed = 121
    random.seed(seed)  # Set Python's random seed
    np.random.seed(seed)  # Set NumPy's random seed
    torch.manual_seed(seed)  # Set PyTorch's CPU random seed
    torch.cuda.manual_seed(seed)  # Set PyTorch's GPU random seed (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # For all GPUs if you have multiple

    
    """Model & training hyper parameters"""
    
    # model configuration for any given run
    context_length = 512
    d_model = 768
    num_heads = 16
    num_hidden_layers = 4
    d_hidden = 2048
    num_encoders = 7 
    num_epochs = 10
    mod_name = 'etpol'

    lr =  6.871199163253075e-05
    pdrop = 0.3099236390195026
    batch_size = 32
    
    # instantiate model with given parameters
    model = ETPOL(
        vocab_size=20000,
        context_length=context_length,
        d_model=d_model,
        d_hidden=d_hidden,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
        num_encoders=num_encoders,
        pdrop=pdrop 
    )


    model_hyperparams = {
        'context_length': context_length,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_hidden_layers': num_hidden_layers,
        'd_hidden': d_hidden,
        'num_encoders': num_encoders, 
        'lr': lr,
        'dropout': pdrop,
        'batch_size': batch_size,
        'num_epochs': num_epochs
    }
    
    # helpful object to keep track of which hyperparameters were used for a given trial run
    model_hyperparams_str = '\n'.join(f'{k}: {v}' for k, v in model_hyperparams.items())
    print(model_hyperparams_str)

    # make sure to make use of gpu if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")
    # good for binary classification, label smoothing of 0.1
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    print('loading tokenizer')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(Path('wp_tokenizer'), model_max_lenth=context_length)

    print('loading df..')
    df=pd.read_csv(Path('data/all_posts.csv'))

    # splitting up the data, making sure to stratify by the label
    print('splitting into train and test sets')
    train_df, temp = train_test_split(df, test_size=0.3, stratify=df['affiliation'])
    val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp['affiliation'])

    print('creating datasets..')
    # wrap df in dataset objects
    train_dataset = PostsDataset(df=train_df, tokenizer=tokenizer, context_length=context_length)
    val_dataset = PostsDataset(df=val_df, tokenizer=tokenizer, context_length=context_length)
    test_dataset = PostsDataset(test_df, tokenizer=tokenizer, context_length=context_length)

    # dataloaders expect dataset objects, this class batches the observations and tokenizes them on the fly
    print('creating dataloaders')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # get training results, see test_utils for a description of this train method
    results = train(
        model=model,
        mod_name=mod_name,
        save_model=True,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,         
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        loss_fn=loss_fn
    )
    
    # evaluate on test dataset
    f1_test, acc_test, loss_test, prec_test, recall_test = evaluate_model(model, test_dataloader, device, loss_fn)

    results = update_results(results, f1_test, acc_test, loss_test, prec_test, recall_test, 'test')
    
    # save results (also in test_utils.py)
    save_results(results, model_hyperparams_str, mod_name)
