import torch
import torch.nn as nn


class LSTMNetwork(nn.Module):
   def __init__(self, codebook_size, embedding_size, hidden_size):
       super(LSTMNetwork, self).__init__()
       self.hidden_size = hidden_size
       self.embedding = nn.Embedding(codebook_size, embedding_size)
       self.lstm = nn.LSTM(embedding_size, hidden_size, 1,
                           batch_first=True)
       self.dense = nn.Linear(hidden_size, codebook_size)

   def forward(self, x, prev_state):
       embed = self.embedding(x)
       output, state = self.lstm(embed, prev_state)
       logits = self.dense(output)

       return logits, state
   def init_hidden_states(self, batch_size):
       return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))