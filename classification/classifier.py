import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from  clip.clip_model import Net1D 

class ResNetECG(nn.Module):
    def __init__(self,
                 num_classes,                 
                 ecg_channels=12, 
                 ):
        super().__init__()

        filter_list = [64,128,256,512]
        self.resnet = Net1D(
                in_channels=ecg_channels, 
                base_filters=64, 
                ratio=1, 
                filter_list=filter_list, 
                m_blocks_list=[2,2,2,3], 
                kernel_size=16, 
                stride=2, 
                groups_width=16,
                verbose=False, 
                use_bn=True,
        )

        self.head = nn.Linear(filter_list[-1], num_classes)

    def forward(self, x):
        # input x: (B, C, L) -> (B, L, C) 
        x = torch.transpose(x, 1, 2)
        features = self.resnet(x) 
        logit = F.softmax(self.head(features), dim=1) 

        return logit 

# Positional Encoding Layer
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=1024):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, embed_size) with positional encodings
        pos_encoding = torch.zeros(max_len, embed_size)
        
        # Create a position tensor [0, 1, 2, ..., max_len-1].unsqueeze(1) -> shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Define the scaling factors for even and odd indices of the positional encoding
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        # Apply the sine to even indices in the embedding dimension
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        
        # Apply the cosine to odd indices in the embedding dimension
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Register the positional encoding as a buffer (not a parameter, but part of the state)
        pos_encoding = pos_encoding.unsqueeze(0)  # Shape (1, max_len, embed_size)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        # x is expected to be of shape (batch_size, seq_len, embed_size)
        seq_len = x.size(1)
        # Add positional encoding to the input embeddings (broadcasting over batch size)
        return x + self.pos_encoding[:, :seq_len, :]

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_size, num_heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention sublayer with residual connection
        attn_output = self.self_attn(x, x, x, mask, need_weights=False)[0]
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-forward sublayer with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x

class TransformerECG(nn.Module):
    def __init__(self, embed_size=256, ecg_length=1024, ecg_channel=12,  num_heads=4, ff_hidden_size=512, num_layers=3, dropout=0.1, num_classes=2):
        super(TransformerECG, self).__init__()
        self.embed_size = embed_size
        self.embedding = nn.Linear(ecg_channel, embed_size) 
        self.positional_encoding = PositionalEncoding(embed_size)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_size, num_heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_size * ecg_length, num_classes) 

    def forward(self, x, mask=None):
        # Add positional encoding to input embeddings
        # (B, L, C) -> (B, L, embed_size)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        
        # Pass input through each transformer encoder layer
        for layer in self.layers:
            x = layer(x, mask)
        
        logit = self.head(torch.flatten(x, 1, -1)) 
        return logit
