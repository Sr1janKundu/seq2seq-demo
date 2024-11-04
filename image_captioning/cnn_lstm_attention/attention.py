import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    """
    def __init__(self, encoder_dim):
        """

        Args:
            encoder_dim ():
        """
        super(Attention, self).__init__()
        self.Q = nn.Linear(512, 512)
        self.K = nn.Linear(encoder_dim, 512)
        self.V = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, img_features, hidden_state):
        """

        Args:
            img_features ():
            hidden_state ():

        Returns:

        """
        Q_h = self.Q(hidden_state).unsqueeze(1)
        K_s = self.K(img_features)
        att = self.tanh(K_s + Q_h)
        e = self.V(att).squeeze(2)
        alpha = self.softmax(e)
        context = (img_features*alpha.unsqueeze(2)).sum(1)

        return context, alpha