from torchcrf import  CRF
import torch

n_tags = 5
model = CRF(num_tags=n_tags, batch_first=True)

seq_length = 2
batch_size = 1
emissions = torch.randn(batch_size, seq_length, n_tags)
tags = torch.tensor([[0, 1]], dtype=torch.long)  # (seq_length, batch_size)
mask = torch.tensor([[1, 1]], dtype=torch.uint8)

score = model(emissions, tags, mask=mask)

model.decode(emissions)