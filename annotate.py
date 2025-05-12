from transformers import PreTrainedTokenizerFast
from models import ETPOL
import pandas as pd
from pathlib import Path
import torch


context_length = 512
d_model = 768
num_heads = 16
num_hidden_layers = 4
d_hidden = 2048
num_encoders = 7 
 
 
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


model.load_state_dict(torch.load('./models/etpol.pth'))
print('model state loaded successfully')



reverse_label_map = {1: 'left', 0: 'right'}

def get_pol(content):
    tokenized = tokenizer.encode(content,
                                 max_length=context_length,
                                 padding='max_length',
                                 truncation = True
                            )
    inp = torch.tensor(tokenized).unsqueeze(0).to(device)
    logits = model(inp)
    return reverse_label_map[int(torch.argmax(logits, dim=-1))]



model.to(device)
posts = pd.read_csv('./scrape/subreddits_cleaned.csv')


print('starting to label')
posts['content'] = posts['content'].fillna('').astype(str)
# loop is working, now hopefully i can to device this shit on the gpu
posts['affiliation'] = posts['content'].map(get_pol)

posts.to_csv('./annotated_subreddit/posts_w_pol.csv')