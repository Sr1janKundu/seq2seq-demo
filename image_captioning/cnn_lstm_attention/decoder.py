import torch
import torch.nn as nn
from attention import Attention
from IPython import embed

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
            img_features (torch.tensor):
            captions ():

        Returns:

        """
        batch_size = img_features.size(0)
        h, c = self.get_init_lstm_state(img_features)

        max_timespan = max([len(caption) for caption in captions]) - 1    # use this if 1d list is passed in place of 2d list
        # to-do: change according to dataset; for coco captions, the loader returns a 2d list - list of 5 lists, each containing batch_size number of captions
        # max_timespan = max(len(caption) for sublist in captions for caption in sublist)
        # maximum number of decoding time steps allowed during caption generation
        # determines the max length of the output sequence (captions) that the model will generate for each image

        prev_words = torch.zeros(batch_size, 1).long().to(img_features.device)
        if self.use_tf:
            # teacher forcing
            embedding = self.embedding(captions) if self.training else self.embedding(prev_words)
            # to-do: change according to dataset exactly like max_timespan
            # EITHER PASS 1D LIST AS captions IN train.py, OR SAMPLE FROM LIST OF 5 IN THE START OF forward()
            # self.training is a built-in pytorch attribute, that is automatically set based on model's current mode.
            # The Decoder class inherits from nn.Module, so it automatically gets this self.training attribute.
        else:
            embedding = self.embedding(prev_words)

        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).to(img_features.device)
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).to(img_features.device)

        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate*context

            if self.use_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(self.dropout(h))

            preds[:, t] = output
            alphas[:, t] = alpha

            if not self.training or not self.use_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))

        return preds, alphas

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

    def caption(self, img_features, beam_size):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        Args:
            img_features (torch.tensor):
            beam_size ():

        Returns:

        """
        prev_words = torch.zeros(beam_size, 1).long()

        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1)
        alphas = torch.ones(beam_size, 1, img_features.size(1))

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(img_features)

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words // output.size(1)        # integer division
            next_word_idxs = top_words % output.size(1)
            # embed()

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > 50:
                break
            step += 1

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]

        return sentence, alpha