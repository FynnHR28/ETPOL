from transformers import PreTrainedTokenizerFast
from models import ETPOL
import pandas as pd
from pathlib import Path
import torch


# the best configuration as discovered in opt.py
context_length = 512
d_model = 768
num_heads = 16
num_hidden_layers = 4
d_hidden = 2048
num_encoders = 7 
 
# set the device to the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr =  6.871199163253075e-05
pdrop = 0.3099236390195026


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

tokenizer = PreTrainedTokenizerFast.from_pretrained(Path('wp_tokenizer'), model_max_lenth=context_length)


# load the saved state of the best ETPOL configuration, only executable in my remote connection to the hpc where model state is saved
model.load_state_dict(torch.load('./models/etpol.pth'))
print('model state loaded successfully')


# same label map used during training
reverse_label_map = {1: 'left', 0: 'right'}

# method to get the english label of the input text
def get_pol(content):
    tokenized = tokenizer.encode(content,
                                 max_length=context_length,
                                 padding='max_length',
                                 truncation = True
                            )
    inp = torch.tensor(tokenized).unsqueeze(0).to(device)
    # pass through model
    logits = model(inp)
    # extract the higher value probability and return the corresponding character label
    return reverse_label_map[int(torch.argmax(logits, dim=-1))]


# make sure model is on gpu
model.to(device)
# read in the scraped subreddit data
posts = pd.read_csv('./scrape/subreddits_cleaned.csv')


print('starting to label')
posts['content'] = posts['content'].fillna('').astype(str)
model.eval()
# label each post using the get_pol method
posts['affiliation'] = posts['content'].map(get_pol)

# save resulting dataframe to csv
posts.to_csv('./annotated_subreddit/posts_w_pol.csv')