import torch


def simple_attention(query, key, value):
    """implements the standard attention mechanism given query, key and value vectors"""
    # check dimensions equal
    assert query.size() == key.size()
    assert query.size(-1) == key.size(-1) == value.size(-1)
    # Attention (Q, K, V) = (Q * K_T)/sqrt(dim) * V
    attn = torch.matmul(query, key.permute(0, 2, 1)) / torch.sqrt(torch.tensor(query.shape[-1]).float())
    w = torch.nn.functional.softmax(attn, dim=-1)  # softmax over attention to obtain weights
    return torch.matmul(w, value)


class SelfAttention(torch.nn.Module):

    def __init__(self, hidden_size, dropout, attn_fun=simple_attention):
        super().__init__()
        # VARIALBE INITIALIZATIONS
        self.attn_fun = attn_fun
        self.hidden_size = hidden_size
        self.dropout = dropout

        # MODEL LAYERS
        self.final_linear_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, query, key, value):
        assert query.size() == key.size() == value.size()  # check all sizes equal
        weighted_vectors = self.attn_fun(query, key, value)
        return self.final_linear_layer(weighted_vectors)


class PositionwiseFeedForward(torch.nn.Module):

    def __init__(self, hidden_dim, ff_dim, dropout):
        super().__init__()
        # VARIABLE INITIALIZATIONS
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.dropout_val = dropout

        # MODEL LAYERS
        # project between model dimension (hidden_dim) and feedforward dimension (ff_dim)
        self.fc1 = torch.nn.Linear(self.hidden_dim, self.ff_dim)
        self.fc2 = torch.nn.Linear(self.ff_dim, self.hidden_dim)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(self.dropout_val)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class SelfAttentionBlock(torch.nn.Module):
    """implements a self-attention block as """

    def __init__(self, max_seq_len, hidden_dim, dropout, positionwise_feedforward_dim,
                 attn_mechanism=SelfAttention, attn_fun=simple_attention):
        super().__init__()
        # VARIABLE INITIALIZATIONS
        self.L = max_seq_len
        self.hidden_dim = hidden_dim
        self.dropout_val = dropout
        self.attn_fun = attn_fun
        self.ff_dim = positionwise_feedforward_dim

        # MODEL LAYERS
        self.attn = attn_mechanism(hidden_size=self.hidden_dim,
                                   dropout=self.dropout_val,
                                   attn_fun=self.attn_fun)
        self.pw_ff = PositionwiseFeedForward(self.hidden_dim, self.ff_dim, self.dropout_val)
        self.ln1 = torch.nn.LayerNorm(normalized_shape=torch.Size((self.L, self.hidden_dim)))
        self.ln2 = torch.nn.LayerNorm(normalized_shape=torch.Size((self.L, self.hidden_dim)))

    def forward(self, q, k, x):
        x_ = x  # store residual value
        x = self.attn(q, k, x)  # apply attention mechanism
        x = self.ln1(x)  # normalize x
        x = self.pw_ff(x)  # apply point-wise feedforward
        x = self.ln2(x)  # normalize x
        x = x + x_  # add residual connection
        return x
