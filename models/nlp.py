import torch
import torch.nn as nn
from transformers import BertModel


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.proj = nn.Linear(embed_dim, d_model)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x, key_padding_mask=None):
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)
        out, _ = self.attn(query, x, x, key_padding_mask=key_padding_mask)
        return self.proj(out.squeeze(1))


class TextEncoder(nn.Module):
    """
    Text encoder using frozen BERT embeddings + learnable positional encoding + attention pooling.
    Input: input_ids (B, seq_len), attention_mask (B, seq_len)
    Output: (token_features, sentence_vector)
        - token_features: (B, seq_len, 768) — token embeddings + positional encoding
        - sentence_vector: (B, d_model) — pooled representation
    """

    def __init__(self, d_model=512, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.embed_dim = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, self.embed_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        self.pool = AttentionPooling(self.embed_dim, d_model)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            embeddings = self.bert.embeddings.word_embeddings(input_ids)

        seq_len = input_ids.shape[1]
        token_features = embeddings + self.pos_encoding[:, :seq_len, :]

        padding_mask = (attention_mask == 0)
        sentence_vector = self.pool(token_features, key_padding_mask=padding_mask)

        return token_features, sentence_vector
