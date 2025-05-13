from random import shuffle
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from pathlib import Path
import csv

# handy method to update the results for each metric within the training loop
def update_results(results, f1, acc, loss, prec, recall, test_type):
    
    if test_type == 'test':
        results[f'f1_{test_type}'] = f1
        results[f'acc_{test_type}'] = acc
        results[f'loss_{test_type}'] = loss
        results[f'prec_{test_type}'] = prec
        results[f'recall_{test_type}'] = recall
    else:
        results[f'f1_{test_type}'].append(f1)
        results[f'acc_{test_type}'].append(acc)
        results[f'loss_{test_type}'].append(loss)
        results[f'prec_{test_type}'].append(prec)
        results[f'recall_{test_type}'].append(recall)
    
    return results

# save the results of training to two files: a csv with per epoch metric performance, and a txt file with the parameter configuration for later reference
def save_results(results, hyper_params_str, title):

    file_path = Path('results/' + title + '_metrics.csv')
    
    with open(file_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'set', 'loss', 'accuracy', 'precision', 'recall', 'f1'])
        writer.writeheader()
        for test_type in ['train', 'val']:
            for i, (f1, acc, loss, prec, recall) in enumerate(
                    zip(results[f'f1_{test_type}'], 
                    results[f'acc_{test_type}'], 
                    results[f'loss_{test_type}'], 
                    results[f'prec_{test_type}'], 
                    results[f'recall_{test_type}']), 1):
                writer.writerow({'epoch': i, 'set': test_type, 'loss': loss, 'accuracy': acc['accuracy'], 'precision': prec, 'recall': recall, 'f1': f1})
        # finish with test performance
        writer.writerow({'epoch': None, 'set': 'test', 'loss': results['loss_test'], 'accuracy': results['acc_test']['accuracy'], 'precision': results['prec_test'], 'recall': results['recall_test'], 'f1': results['f1_test']})

    # save hyperparams as a string
    with open('results/' + title + '_params.txt', 'w') as f:
        f.write(title + '\nParams:\n' + hyper_params_str + '\n\n')
        
        

def evaluate_model(model, dataloader, device, loss_fn):
    """Takes the model and a dataset. Evaluates the model on the dataset, printing out overall accuracy."""
    # NOTE to make it simple, dataset is a dataloader already
    metric = evaluate.load("accuracy")
    total_loss = 0
    model.eval()
    for batch in dataloader:
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            logits = model(input_ids)
            loss = loss_fn(logits, labels)  # Compute loss
            total_loss += loss.item()

        predictions = torch.argmax(logits, dim=-1)
        # Flatten predictions and labels 
        predictions = predictions.view(-1).detach().cpu().numpy()  # Shape: [batch_size * seq_len]
        labels = labels.view(-1).detach().cpu().numpy()  # Shape: [batch_size * seq_len]

        metric.add_batch(predictions=predictions, references=labels)
    
    # compute all metric
    # average = 'micro' uses a global count of the total TPs, FNs and FPs.
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(y_true=labels, y_pred=predictions, average='micro')
    
    prec = precision_score(y_true=labels, y_pred=predictions, average='micro')
    recall = recall_score(y_true=labels, y_pred=predictions, average='micro')
    
    acc = metric.compute()
    
    return f1, acc, avg_loss, prec, recall

  

def train(model, train_dataloader, val_dataloader, num_epochs, 
          lr, device, loss_fn, mod_name, save_model):
    # advanced optimizer for how to update model weights
    # often results in better generalization than SGD
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # results dictionary initialized
    results = {
        'f1_train': [],
        'acc_train': [],
        'loss_train': [],
        'prec_train': [],
        'recall_train': [],
        'f1_val': [],
        'acc_val': [],
        'loss_val': [],
        'prec_val': [],
        'recall_val': [],
        'f1_test': 0,
        'acc_test': 0,
        'loss_test': 0,
        'prec_test': 0,
        'recall_test': 0
    }   
    # An object that adjusts the learning rate on the fly, allows for 
    # quicker learning at the beginning of an epoch. Slowly reducing it after
    # the warmup steps results in better convergence
    total_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = int(total_steps*0.1)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        # first 10% of training steps ramps lr up to set lr, then it decays
        # over the rest of the steps for better convergence
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps-num_warmup_steps
    )
    
    model.to(device)
     
    model.train()
    #est_val_loss = float('inf')
    patience = 0
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch}")
        epoch_loss = []
        for batch in tqdm(train_dataloader, unit='batch'):
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # outputs our class probs & get loss
            logits = model(input_ids)
            
            loss = loss_fn(logits, labels)
            # update epoch loss
            epoch_loss.append(loss.detach())
            
            # backprop
            loss.backward()

            optimizer.step()
            # update learning rate
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # calculate all metrics
        avg_train_loss = np.mean([loss.cpu().item() for loss in epoch_loss])
        # f1, acc, avg_loss, prec, recall in that order
        f1_train, acc_train, _, prec_train, recall_train = evaluate_model(model, train_dataloader, device, loss_fn)
        results = update_results(results, f1_train, acc_train, avg_train_loss, prec_train, recall_train, 'train')
        
        f1_val, acc_val, loss_val, prec_val, recall_val = evaluate_model(model, val_dataloader, device, loss_fn)
        
        # early stopping mechanism, used for efficiency and stopping poor configurations early 
        if loss_val > best_val_loss:
            patience += 1
            if patience > 1: 
                print('stopping training early! (val loss got worse twice in a row)')
                break
        else:
            best_val_loss = loss_val
            patience = 0
        
        # update the results dictionary and print 
        results = update_results(results, f1_val, acc_val, loss_val, prec_val, recall_val, 'val')
        print(results)

    # if I wanted to save the model (like for the best config I found) this saves its state to a .pth file
    if save_model:
        torch.save(model.state_dict(), 'models/' + mod_name + '.pth')
    
    # the results dictionary is returned at the end of training, and it is saved in the main.py script!
    return results
    
    



