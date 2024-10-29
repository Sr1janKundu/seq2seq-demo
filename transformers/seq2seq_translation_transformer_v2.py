import math
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                         fields=(german, english),
                                                         root='.data')

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_len=5000, device="cuda"):
        super(PositionalEncoding, self).__init__()
        self.device = device
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, embedding_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, embedding_size)
        return x + self.pe[:x.size(0), :]
    

class Transformer(nn.Module):
    def __init__(self,
                 embedding_size,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_size,
                 num_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 forward_expansion,
                 dropout,
                 max_len,
                 device):
        super(Transformer, self).__init__()

        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        
        # Replace learned positional embeddings with sinusoidal encodings
        self.src_positional_encoding = PositionalEncoding(embedding_size, max_len, device)
        self.trg_positional_encoding = PositionalEncoding(embedding_size, max_len, device)
        
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_size

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask
    
    def forward(self, src, trg):
        # src shape: (src_len, N), trg shape: (trg_len, N)
        
        # Create word embeddings
        src_embedded = self.src_word_embedding(src)  # (src_len, N, embedding_size)
        trg_embedded = self.trg_word_embedding(trg)  # (trg_len, N, embedding_size)
        
        # Add positional encoding
        src_embedded = self.src_positional_encoding(src_embedded)
        trg_embedded = self.trg_positional_encoding(trg_embedded)
        
        # Apply dropout
        src_embedded = self.dropout(src_embedded)
        trg_embedded = self.dropout(trg_embedded)
        
        # Create masks
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.shape[0]).to(self.device)

        # Pass through transformer
        out = self.transformer(
            src_embedded,
            trg_embedded,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask
        )

        # Project to vocabulary size
        out = self.fc_out(out)
        return out
    

# setup training phase
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 100
learning_rate = 3e-4
batch_size = 64

# Model hyperparameters 
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device
).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Tensorboard
writer = SummaryWriter("runs_V2/loss_plot")
step = 0

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size = batch_size,
    sort_within_batch = True,
    sort_key = lambda x:len(x.src),
    device = device)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "ein pferd geht unter einer brücke neben einem boot."
# google translate to English: "A horse walks under a bridge next to a boat."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch+1} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    progress_bar = tqdm(enumerate(train_iterator), total=len(train_iterator), desc=f'Epoch {epoch + 1}')

    for batch_idx, batch in progress_bar:
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target[:-1, :])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    print(f"Current learning rate: {scheduler.get_last_lr()[0]}")
    scheduler.step(mean_loss)
    
    #calculate validation loss every 100 epoch
    if epoch % 10 == 0:
        score = bleu(validation_data[1:100], model, german, english, device)
        print(f"Bleu score on validation: {score * 100:.2f}")

# running on entire test data 
score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score on test: {score * 100:.2f}")