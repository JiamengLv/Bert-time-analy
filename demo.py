import torch
from bert import BERT
from fc import FC


seq_len = 5
hidden = 30
out_emb_size = 8

device = "cpu"

bert_model = BERT(seq_len, hidden=hidden, n_layers=4, attn_heads=1, dropout=0.1)
fc_model = FC(hidden, out_emb_size).to(device)

input = torch.zeros(size=(1,seq_len,hidden))

enn = bert_model(input)
out = fc_model(enn)

print(out)

