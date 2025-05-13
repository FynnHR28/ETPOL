from models import ETPOL
from pathlib import Path
import torch

context_length = 512
d_model = 512

model = ETPOL(
    vocab_size=20000,
    context_length=context_length,
    d_model=d_model,
    num_heads=8,
    num_hidden_layers=2,
    d_hidden=2048,
    num_encoders=8
)

fake_input = torch.ones(4, context_length, dtype=torch.int)

output = model(fake_input)
print(output.shape)
print(output)
