import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils_mod import translate_sentence, bleu, modified_bleu_score, save_checkpoint, load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Multi30k.download(root='.data')

spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(tokenize=tokenizer_ger, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenizer_eng, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                         fields=(german, english),
                                                         root='.data')

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=True)

    def forward(self, x):
        embeddings = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embeddings)

        # Combine forward and backward hidden states
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        
        cell = cell.view(self.num_layers, 2, -1, self.hidden_size)
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)

        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (N) or (N, 1)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        else:
            x = x.transpose(0, 1)  # Make it (1, N)
        
        embedded = self.dropout(self.embedding(x))
        # embedded shape: (1, N, embedding_size)

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output shape: (1, N, hidden_size)

        prediction = self.fc(output.squeeze(0))
        # prediction shape: (N, output_size)

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_force_ratio=0.5, beam_width=3, max_length=50):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.output_size

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(source)

        # First input to the decoder is the <sos> token
        input_seq = target[0]

        for t in range(1, target_len):
            if random.random() < teacher_force_ratio:
                # Teacher forcing: use the ground-truth target as the next input
                output, hidden, cell = self.decoder(input_seq, hidden, cell)
                outputs[t] = output
                input_seq = target[t]
            else:
                # Use the previous output as the next input
                output, hidden, cell = self.decoder(input_seq, hidden, cell)
                outputs[t] = output
                # Get the highest predicted token
                top1 = output.argmax(1)
                input_seq = top1

        return outputs

# Training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Beam Search parameters
beam_width = 3
max_length = 50

writer = SummaryWriter(f'runs_mod/loss_plot')
step = 0

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size * 2, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint_mod.pth.tar'), model, optimizer)

sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

for epoch in range(num_epochs):
    print(f'Epoch {epoch} / {num_epochs}')
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(model, sentence, german, english, device, beam_width, max_length)
    print(f'Translated example sentence \n {translated_sentence}')

    model.train()

    progress_bar = tqdm(enumerate(train_iterator), total=len(train_iterator), desc=f'Epoch {epoch + 1}')

    for batch_idx, batch in progress_bar:
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target, teacher_force_ratio=0.5, beam_width=beam_width, max_length=max_length)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    score = bleu(test_data[1:100], model, german, english, device, beam_width, max_length)
    # score = modified_bleu_score()
    print(f"Bleu score {score * 100:.2f}")