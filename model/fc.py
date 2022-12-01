import torch.nn as nn

class FC(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, out_emb_size):

        """
        :param hidden: output size of BERT model
        :param vocab_size:
        """
        super().__init__()
        self.linear = nn.Linear(hidden, out_emb_size)

    def forward(self, x):
        return self.linear(x)
