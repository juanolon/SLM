import copy

from torch import nn
import torch.nn.functional as F
from models.promoter_model import GaussianFourierProjection, Dense


class CNNModel(nn.Module):
    def __init__(self, alphabet_size, num_cls, num_cnn_stacks, classifier=False):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.classifier = classifier
        self.num_cls = num_cls
        
        self.clean_data = classifier
        self.cls_expanded_simplex = False
        self.hidden_dim = 128
        self.mode = 'new_diff'
        self.dropout = 0.0
        self.cls_free_guidance = True
        self.num_cnn_stacks = num_cnn_stacks

        if self.clean_data:
            self.linear = nn.Embedding(self.alphabet_size, embedding_dim=self.hidden_dim)
        else:
            expanded_simplex_input = self.cls_expanded_simplex or not classifier and (self.mode == 'dirichlet' or self.mode == 'riemannian')
            inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1)
            if (self.mode == 'ardm' or self.mode == 'lrar') and not classifier:
                inp_size += 1 # plus one for the mask token of these models
            self.linear = nn.Conv1d(inp_size, self.hidden_dim, kernel_size=9, padding=4)
            self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim= self.hidden_dim),nn.Linear(self.hidden_dim, self.hidden_dim))

        self.num_layers = 5 * self.num_cnn_stacks
        self.convs = [nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=64, padding=256)]
        self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(self.num_cnn_stacks)])
        self.time_layers = nn.ModuleList([Dense(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.final_conv = nn.Sequential(nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv1d(self.hidden_dim, self.hidden_dim if classifier else self.alphabet_size, kernel_size=1))
        self.dropout = nn.Dropout(self.dropout)
        if classifier:
            self.cls_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_dim, self.num_cls))

        if self.cls_free_guidance and not self.classifier:
            self.cls_embedder = nn.Embedding(num_embeddings=self.num_cls + 1, embedding_dim=self.hidden_dim)
            self.cls_layers = nn.ModuleList([Dense(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
    def forward(self, seq, t, cls = None, return_embedding=False):
        if self.clean_data:
            feat = self.linear(seq)
            feat = feat.permute(0, 2, 1)
        else:
            time_emb = F.relu(self.time_embedder(t))
            feat = seq.permute(0, 2, 1)
            feat = F.relu(self.linear(feat))

        if self.cls_free_guidance and not self.classifier:
            cls_emb = self.cls_embedder(cls)

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            if self.cls_free_guidance and not self.classifier:
                h = h + self.cls_layers[i](cls_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        if self.classifier:
            feat = feat.mean(dim=1)
            if return_embedding:
                embedding = self.cls_head[:1](feat)
                return self.cls_head[1:](embedding), embedding
            else:
                return self.cls_head(feat)
        return feat
