import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        weights = Inception_V3_Weights.DEFAULT
        self.inception = models.inception_v3(weights=weights)
        # self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        for param in self.inception.parameters():
            param.requires_grad = train_CNN
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # Set model to eval mode if not training CNN
        if not self.train_CNN:
            self.inception.eval()

        # Forward pass with no aux_logits
        features = self.inception(images)

        # for name, param in self.inception.named_parameters():
        #     if "fc.weight" in name or "fc.bias" in name:        # last modified layer weights finetuning
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = self.train_CNN
        # If the output is a tuple (happens when aux_logits=True)
        if isinstance(features, tuple):
            features = features[0]  # Get main output, ignore aux output
            
        return self.dropout(self.relu(features))
        
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim = 0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs
    
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        
        return outputs
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None   # hidden and cell state initialization for LSTM

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)        # take word with highest prob

                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]