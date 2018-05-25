import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, hidden_dim, output_size):

        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.2)


    def init_hidden(self):
        return(autograd.Variable(torch.randn(1, 1, self.hidden_dim)),
                        autograd.Variable(torch.randn(1, 1, self.hidden_dim)))


    def forward(self, batch, lengths):

        self.hidden = self.init_hidden()

        packed_input = pack_padded_sequence(batch, lengths, batch_first=True)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)

        output = self.hidden2out(ht[-1])
        output = self.softmax(output)

        return output
