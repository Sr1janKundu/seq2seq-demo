import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed
from tensorflow.python.layers.core import dropout
from torch.nn.functional import embedding
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter

spacy_ger = spacy.load('de')        # German tokenizer
spacy_eng = spacy.load('en')        # English tokenizer

def tokenizer_ger(text):
    """
    Takes text in German, tokenizes and returns list of tokens

    :param text: amy text in German
    :return: list of tokens from input text
    """
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenizer_eng(text):
    """
    Takes text in English, tokenizes and returns list of tokens

    :param text: amy text in English
    :return: list of tokens from input text
    """
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# define preprocessing
german = Field(tokenize=tokenizer_ger,
               lower=True,
               init_token='<sos>',
               eos_token='<eos>')
english = Field(tokenize=tokenizer_eng,
               lower=True,
               init_token='<sos>',
               eos_token='<eos>')

# dataset
train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                         fields=(german, english))
# vocabulary
german.build_vocab(train_data, max_size = 10000, min_freq = 2)
english.build_vocab(train_data, max_size = 10000, min_freq = 2)

# model
class Encoder(nn.Module):
    def __init__(self, input_size:int, embedding_size:int, hidden_size:int, num_layers:int, p:float):
        """
        Initialize the Encoder class.

        :param input_size: int - Size of the input vocabulary; for example, the size of the German vocabulary
        :param embedding_size: int - Dimension of the embedding space to map each word into, where each word is represented by a d-dimensional vector
        :param hidden_size: int - The number of features in the hidden state of the LSTM
        :param num_layers: int - Number of LSTM layers in the encoder
        :param p: float - Dropout rate applied to the outputs of the LSTM layers to prevent overfitting
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        """
        Define the forward method
        :param x: The vector of indices of the tokenized input sentence that map to actual words in the vocabulary
                shape: (seq_length, N); N is batch size
        :return:
        """
        embeddings = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embeddings)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size:int, embedding_size:int, hidden_size:int, output_size:int, num_layers:int, p:float):
        """
        Initialize the Decoder class.

        :param input_size: int - Size of the input vocabulary; for example, the size of the English vocabulary
        :param embedding_size: int - Dimension of the embedding space to map each word into, where each word is represented by a d-dimensional vector
        :param hidden_size: int - The number of features in the hidden state of the LSTM
        :param output_size: int - Same size as the input size
        :param num_layers: int - Number of LSTM layers in the encoder
        :param p: float - Dropout rate applied to the outputs of the LSTM layers to prevent overfitting
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        """
        Define the forward method
        :param x: A single word
        :param hidden:
        :param cell:
        :return:
        """
        # shape of x: (N), but we want (1, N), so that N batches of a single word
        x = x.unsqueeze(0)

        embeddings = self.dropout(self.embedding(x))
        # embedding shape = (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embeddings, (hidden, cell))
        # shape of outputs (1, N, hidden_size)

        predictions = self.fc(outputs)
        # shape of predictions = (1, N, length_of_vocab)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    pass