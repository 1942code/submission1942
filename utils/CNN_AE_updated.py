import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm + 1e-7)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=4, head_dim=32):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_proj = nn.Linear(input_dim, 3 * num_heads * head_dim)
        self.output_proj = nn.Linear(num_heads * head_dim, input_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.output_proj(attn_output)
    
class CNN_AE_Classifier(nn.Module):
    def __init__(self, input_dim, num_class, low_dim=12):
        super(CNN_AE_Classifier, self).__init__()

        self.attention = MultiHeadAttention(input_dim=120, num_heads=4, head_dim=32)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, 1, 2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 64, 5, 1, 2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2, 2, 0),
            nn.Conv1d(64, 128, 5, 1, 2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2, 2, 0),
            nn.Flatten(),
            nn.Linear(3840, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 3840),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 30)),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.ConvTranspose1d(32, 1, 5, 1, 2),
            nn.Tanh()
        )

        self.classifier = nn.Linear(128, num_class)
        self.l2norm = Normalize(2)

        # projection MLP
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, low_dim)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(2) 
        
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        
        x = self.attention(x)

        feat_ = self.encoder(x)
        
        x_ = self.decoder(feat_)
        
        out = self.classifier(feat_)
        
        feat = F.relu(self.fc1(feat_))
        feat = self.fc2(feat)
        feat = self.l2norm(feat)
        
        return out, feat, feat_, x_

# 使用示例
if __name__ == "__main__":
    batch_size = 64
    in_dim = 120
    num_class = 2
    
    model = CNN_AE_Classifier(in_dim, num_class)
    
    x = torch.randn(batch_size, 1, in_dim)
    
    out, feat, feat_, x_ = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape (classification): {out.shape}")
    print(f"Low-dim feature shape: {feat.shape}")
    print(f"High-dim feature shape: {feat_.shape}")
    print(f"Reconstructed input shape: {x_.shape}")
