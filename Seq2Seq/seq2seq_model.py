# from os import write

import torch
import torch.nn as nn
import torch.optim as optim
# from IPython import embed
# from tensorflow.python.layers.core import dropout
# from tensorflow.python.training.checkpoint_utils import load_checkpoint
# from torch.nn.functional import embedding
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
# import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
Multi30k.download(root='.data')


# run in terminal `python -m spacy download de_core_news_sm`
spacy_ger = spacy.load("de_core_news_sm")        # German tokenizer
# run in terminal `python -m spacy download en_core_web_sm`
spacy_eng = spacy.load("en_core_web_sm")        # English tokenizer

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
                                                         fields=(german, english),
                                                         root='.data')
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
        # x shape: (seq_length, N) where N is batch size

        embeddings = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embeddings)
        # outputs shape: (seq_length, N, hidden_size)

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

        self.dropout = nn.Dropout(p)
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
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)

        embeddings = self.dropout(self.embedding(x))
        # embedding shape = (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embeddings, (hidden, cell))
        # shape of outputs (1, N, hidden_size)

        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just remove the first dim
        predictions = self.fc(outputs)
        # shape of predictions = (1, N, length_of_vocab)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        """

        :param encoder:
        :param decoder:
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        """

        :param source:
        :param target:
        :param teacher_force_ratio:
        :return:
        """
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # Grab start token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output
            # output shape: (N, eng_vocab_size)

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different from what the
            # network is used to.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

# training hyperparameters
num_epochs = 50
learning_rate = 0.001
batch_size = 64

# model hyperparameters
load_model = False
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024      # Needs to be the same for both RNNs'
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size = batch_size,
    sort_within_batch = True,
    sort_key = lambda x:len(x.src),
    device = device)

encoder_net = Encoder(input_size_encoder,
                      encoder_embedding_size,
                      hidden_size,
                      num_layers,
                      enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder,
                      decoder_embedding_size,
                      hidden_size,
                      output_size,
                      num_layers,
                      dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

pad_idx = english.vocab.stoi['<pad>']       # string to index

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

# score = bleu(test_data, model, german, english, device)       # if load model
sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

for epoch in range(num_epochs):
    print(f'Epoch {epoch} / {num_epochs}')
    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(model, sentence, german, english, device, max_length=50)
    print(f'Translated example sentence \n {translated_sentence}')

    model.train()

    progress_bar = tqdm(enumerate(train_iterator), total=len(train_iterator), desc=f'Epoch {epoch + 1}')

    # for batch_idx, batch in enumerate(train_iterator):
    for batch_idx, batch in progress_bar:
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target)
        # output shape: (trg_len, batch_size, output_dim)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshaping. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)   # to keep the gradients within healthy range

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    score = bleu(test_data[1:100], model, german, english, device)
    print(f"Bleu score {score * 100:.2f}")