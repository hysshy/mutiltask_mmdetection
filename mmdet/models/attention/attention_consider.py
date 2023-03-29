import torch
import torch.nn as nn

# Define the attention condenser module
class AttentionCondenser(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(AttentionCondenser, self).__init__()
        # A convolution layer to reduce the channel dimension
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        # A softmax layer to normalize the attention matrix
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x = list(x)
            for i in range(len(x)):
                # Get the batch size, channel number, height and width of the input feature map
                b, c, h, w = x[i].size()
                # Reshape the input feature map to (b, c, h*w)
                x[i] = x[i].view(b, c, -1)
                # # Transpose the input feature map to (b, h*w, c)
                # x[i] = x[i].permute(0, 2, 1)
                # Apply the convolution layer to reduce the channel dimension to (b, h*w, out_channels)
                x[i] = self.conv(x[i])
                # Compute the attention matrix by multiplying x with its transpose (b, h*w, h*w)
                A = torch.bmm(x[i], x[i].permute(0, 2, 1))
                # Normalize the attention matrix by softmax
                A = self.softmax(A)
                # Apply the attention matrix to the input feature map (b, c, h*w)
                x[i] = torch.bmm(x[i].permute(0, 2, 1), A)
                # Reshape the output feature map to (b, c, h, w)
                x[i] = x[i].view(b, c, h, w)
        return x