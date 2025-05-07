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
from test_utils import train 
import numpy as np
import optuna


seed = 121
random.seed(seed)  # Set Python's random seed
np.random.seed(seed)  # Set NumPy's random seed
torch.manual_seed(seed)  # Set PyTorch's CPU random seed
torch.cuda.manual_seed(seed)  # Set PyTorch's GPU random seed (if using CUDA)
torch.cuda.manual_seed_all(seed)  # For all GPUs if you have multiple
context_length = 512



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device: {device}")

    

print('loading tokenizer')
tokenizer = PreTrainedTokenizerFast.from_pretrained(Path('wp_tokenizer'), model_max_lenth=context_length)

print('loading df..')
df=pd.read_csv(Path('data/all_posts.csv'))
 
 

print('splitting into train and test sets')
train_df, temp = train_test_split(df, test_size=0.3, stratify=df['affiliation'])
val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp['affiliation'])

print('creating datasets..')
train_dataset = PostsDataset(df=train_df, tokenizer=tokenizer, context_length=context_length)
val_dataset = PostsDataset(df=val_df, tokenizer=tokenizer, context_length=context_length)


def objective(trial):
    print("NEXT TRIAL")
    num_epochs = 4
    context_length = 512
    loss_fn = nn.CrossEntropyLoss()
    
     
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 6, 8, 16])
    d_model = trial.suggest_int('d_model', num_heads * 8, num_heads * 64, step=num_heads * 8)
     
    
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 2, 6)
    d_hidden = trial.suggest_categorical('d_hidden',[  128, 256, 512, 1024, 2048])
    num_encoders = trial.suggest_int('num_encoders', 1, 8)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    pdrop = trial.suggest_float('p_drop',0.2, 0.5)
    
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
    
    model_hyperparams_str = '\n'.join(f'{k}: {v}' for k, v in model_hyperparams.items())
    print(model_hyperparams_str)

    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
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

     
    results = train(
        model=model,
        mod_name="opt",
        save_model=False,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        loss_fn=loss_fn
    )
    
    # Extract validation accuracy and loss
    val_acc = results['acc_val'][-1]['accuracy']
    val_loss = results['loss_val'][-1]  # Assuming the last validation loss is logged

    # Log additional metrics for analysis
    trial.set_user_attr("val_loss", val_loss)
    trial.set_user_attr("val_acc", val_acc)
    
    return val_acc




print("STARTING STUDY")
study = optuna.create_study(study_name = 'ETPOL hyperparameter optimization')
study.optimize(objective, n_trials=10)
print("Best trial:")
best_trial = study.best_trial

print(f"  Value: {best_trial.value}")  # Best validation accuracy
print(f"  Params: {best_trial.params}")  # Best hyperparameters
print(f"  Validation Loss: {best_trial.user_attrs['val_loss']}")  # Logged validation loss
print(f"  Validation Accuracy: {best_trial.user_attrs['val_acc']}")  # Logged validation accuracy

