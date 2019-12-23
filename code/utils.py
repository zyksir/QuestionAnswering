import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,dropout,batch_first=True,bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        self.batch_first = batch_first
        return
    def forward(self,input,input_length):
        input_length_sorted = sorted(input_length, reverse=True)
        sort_index = np.argsort(-np.array(input_length)).tolist()
        input_sorted = Variable(torch.zeros(input.size())).cuda()
        batch_size = input.size()[0]
        for b in range(batch_size):
            input_sorted[b,...] = input[sort_index[b],...]
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_sorted,
                                                         input_length_sorted,
                                                         batch_first=self.batch_first)
        output,hidden = self.lstm(packed)
        output, output_length = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)
        # hidden, hidden_length = torch.nn.utils.rnn.pad_packed_sequence(hidden,batch_first=self.batch_first)
        output_resorted = Variable(torch.zeros(output.size())).cuda()
        # hidden_resorted = Variable(torch.zeros(hidden.size())).cuda()
        for b in range(batch_size):
            output_resorted[sort_index[b],...] = output[b,...]
            # hidden_resorted[sort_index[b],...] = hidden[b,...]

        # return output_resorted,hidden_resorted

        return output_resorted