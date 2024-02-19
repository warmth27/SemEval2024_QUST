from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch import nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, embedding_length):
        super(SelfAttention, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        --------

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        # self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
        self.dropout = 0.8
        self.bilstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        self.W_s1 = nn.Linear(2 * hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30 * 2 * hidden_size, 2000)
        self.label = nn.Linear(2000, output_size)

    def attention_net(self, lstm_output):

        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
        encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
        the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
        connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
        pos & neg.
        Arguments
        ---------
        lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
        ---------
        Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
                  attention to different parts of the input sentence.
        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                      attn_weight_matrix.size() = (batch_size, 30, num_seq)
        """
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, input_sentences, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.

        """

        # input = self.word_embeddings(input_sentences)
        input = input_sentences
        input = input.permute(1, 0, 2)
        # if batch_size is None:
        #     h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        #     c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))
        # else:
        #     h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
        #     c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))

        output, (h_n, c_n) = self.bilstm(input)
        output = output.permute(1, 0, 2)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        # h_n.size() = (1, batch_size, hidden_size)
        # c_n.size() = (1, batch_size, hidden_size)
        attn_weight_matrix = self.attention_net(output)
        # attn_weight_matrix.size() = (batch_size, r, num_seq)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)
        # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.
        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        logits = self.label(fc_out)
        # logits.size() = (batch_size, output_size)

        return logits


class RCNN(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size,  embedding_length):
        super(RCNN, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length

        self.dropout = 0.8
        self.lstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
        self.W2 = nn.Linear(2 * hidden_size + embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        """

        The idea of the paper "Recurrent Convolutional Neural Networks for Text Classification" is that we pass the embedding vector
        of the text sequences through a bidirectional LSTM and then for each sequence, our final embedding vector is the concatenation of 
        its own GloVe embedding and the left and right contextual embedding which in bidirectional LSTM is same as the corresponding hidden
        state. This final embedding is passed through a linear layer which maps this long concatenated encoding vector back to the hidden_size
        vector. After this step, we use a max pooling layer across all sequences of texts. This converts any varying length text into a fixed
        dimension tensor of size (batch_size, hidden_size) and finally we map this to the output layer.
        """
        input = input_sentence
        input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        # if batch_size is None:
        #     h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))  # Initial hidden state of the LSTM
        #     c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size))  # Initial cell state of the LSTM
        # else:
        #     h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
        #     c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))

        output, (final_hidden_state, final_cell_state) = self.lstm(input)

        final_encoding = torch.cat((output, input), 2).permute(1, 0, 2)
        y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size()[2])  # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)
        logits = self.label(y)

        return logits

class Simple_CNN(nn.Module):
    def __init__(self, num_units=256, nonlin=nn.ReLU()):
        super().__init__()
        self.window_sizes = [2, 3, 4, 5, 6]
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=768,
                                    out_channels=256,
                                    kernel_size=h),
                          nn.BatchNorm1d(num_features=256),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=512 - h + 1))
            for h in self.window_sizes
        ])
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=256 * len(self.window_sizes),
                            out_features=3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):

        X = X.permute(0, 2, 1)

        out = [conv(X) for conv in self.convs]

        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = self.drop(out)
        out = self.fc(out)
        return out


class LSTM_ATT(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, embedding_length):
        super(LSTM_ATT, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        --------

        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length

        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):

        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input_sentences, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """
        input = input_sentences

        input = input.permute(1, 0, 2)
        # if batch_size is None:
        #     h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        #     c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        # else:
        #     h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        #     c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))

        output, (final_hidden_state, final_cell_state) = self.lstm(input)  # final_hidden_state.size() = (1, batch_size, hidden_size)

        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits