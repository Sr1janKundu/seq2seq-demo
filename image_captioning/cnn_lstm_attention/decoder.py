import torch
import torch.nn as nn
from attention import Attention


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim, tf=False):
        """

        Args:
            vocabulary_size ():
            encoder_dim ():
            tf (boolean): Teacher forcing.
        """
        super(Decoder, self).__init__()
        self.use_tf = tf

        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, 512)
        self.init_c = nn.Linear(encoder_dim, 512)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(512, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(512, vocabulary_size)
        self.dropout = nn.Dropout()

        self.attention = Attention(encoder_dim)
        self.embedding = nn.Embedding(vocabulary_size, 512)
        self.lstm = nn.LSTMCell(512+encoder_dim, 512)

    def forward(self, img_features, captions):
        """

        We can use teacher forcing during training. For reference, refer to
        https://www.deeplearningbook.org/contents/rnn.html
        Args:
            img_features ():
            captions ():

        Returns:

        """
        batch_size = img_features.size(0)
        h, c = self.get_init_lstm_state(img_features)

    def get_init_lstm_state(self, img_features):
        """

        Args:
            img_features ():

        Returns:

        """
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c