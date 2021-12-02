import torch

from constants import PADDING_ITEM
from attention import SelfAttentionBlock


class Caser(torch.nn.Module):

    def __init__(self, args):
        super(Caser, self).__init__()

        # VARIABLE INITIALIZATIONS
        self.d = args.hidden_dim  # embedding dimension
        self.out_size = args.out_size  # corresponds to the number of items
        self.L = args.max_seq_len  # length of the item sequence (i.e., L)
        self.h_filters = args.caser_horizontal_filters

        # MODEL LAYERS
        self.emb_layer = torch.nn.Embedding(num_embeddings=self.out_size + 1, embedding_dim=self.d, max_norm=None,
                                            padding_idx=PADDING_ITEM)
        self.h_convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1,
                                out_channels=1,
                                # kernel_size = (filter_size, d)
                                kernel_size=(f, self.d)),
                torch.nn.ReLU()) for f in self.h_filters])
        self.v_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=1,
                            # kernel_size = (L, 1), where L = number of inputs in sequence
                            kernel_size=(self.L, 1), stride=1),
            torch.nn.ReLU()
        )
        self.fc1 = torch.nn.Linear(self.d + len(self.h_filters), self.out_size)
        self.fc2 = None  # only one layer is used, user information is not included in this model version
        self.dropout = torch.nn.Dropout(args.dropout)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.emb_layer(x)
        x = x.unsqueeze(1)  # conv layer expects [batch x channel x height x width] inputs, unsqueeze channel dim
        x_h = [conv(x) for conv in self.h_convs]
        x_h = [torch.nn.functional.max_pool1d(x_.squeeze(-1), kernel_size=self.L - self.h_filters[i] + 1)
               for i, x_ in enumerate(x_h)]
        x_h = torch.cat(x_h, dim=1).squeeze(dim=-1)
        x_v = self.v_conv(x).squeeze(dim=1).squeeze(dim=1)
        x = torch.cat([x_h, x_v], dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        return self.softmax(x)


class NextItNet(torch.nn.Module):
    class NextItResidualBlockB(torch.nn.Module):
        """implements the residual block B as used in NextItNet"""

        def __init__(self, dilation, dilated_channels, hidden_dim, kernel_size, max_seq_len):
            super().__init__()
            # VARIABLE INITIALIZATIONS
            self.hidden_dim = hidden_dim
            self.dilation = dilation
            self.kernel_size = kernel_size
            self.L = max_seq_len
            self.dilated_channels = dilated_channels

            # MODEL LAYERS
            # (padding_left, padding_right, padding_top, padding_bottom)
            self.pad1 = torch.nn.ConstantPad2d(padding=(0, 0, 0, (self.kernel_size - 1) * self.dilation),
                                               value=PADDING_ITEM)
            # batch_size x 1 x L x hidden_dim (hidden_dim => channels, L => filter dimension)
            self.dconv1 = torch.nn.Conv1d(in_channels=self.hidden_dim,
                                          out_channels=self.dilated_channels,
                                          kernel_size=self.kernel_size,
                                          dilation=self.dilation)
            self.norm1 = torch.nn.LayerNorm(normalized_shape=torch.Size((self.dilated_channels, self.L)))

            self.pad2 = torch.nn.ConstantPad2d(padding=(0, 0, 0, (self.kernel_size - 1) * 2 * self.dilation),
                                               value=PADDING_ITEM)
            self.dconv2 = torch.nn.Conv1d(in_channels=self.dilated_channels,
                                          out_channels=self.hidden_dim,
                                          kernel_size=self.kernel_size,
                                          dilation=2 * self.dilation)
            self.norm2 = torch.nn.LayerNorm(normalized_shape=torch.Size((self.hidden_dim, self.L)))

        def forward(self, x):
            x_ = x  # store residual inputs

            # first block
            x = self.pad1(x)
            x = x.permute(0, 2, 1)  # permute needed as nn.Conv1d expects (batch_size, channels_in, lenght_in) as input
            x = self.dconv1(x)
            x = self.norm1(x)
            x = torch.nn.functional.relu(x)
            x = x.permute(0, 2, 1)  # undo previous permute

            # second block
            x = self.pad2(x)
            x = x.permute(0, 2, 1)  # permute needed as nn.Conv1d expects (batch_size, channels_in, lenght_in) as input
            x = self.dconv2(x)
            x = self.norm2(x)
            x = torch.nn.functional.relu(x)
            x = x.permute(0, 2, 1)  # undo previous permutation to match output shape

            return x + x_  # residual connection

    def __init__(self, args):
        super().__init__()
        # VARIABLE INITIALIZATIONS
        self.hidden_dim = args.hidden_dim
        self.out_size = args.out_size  # corresponds to the number of items
        self.dilated_channels = args.nextitnet_dilated_channels
        self.dilations = args.nextitnet_dilations
        self.kernel_size = args.nextitnet_kernel_size
        self.L = args.max_seq_len

        # MODEL LAYERS
        # one embedding per item => num_embeddings equal to out_size
        self.emb_layer = torch.nn.Embedding(num_embeddings=self.out_size, embedding_dim=self.hidden_dim, max_norm=None,
                                            padding_idx=PADDING_ITEM)
        self.dilated_convolutions = torch.nn.ModuleList([
            self.NextItResidualBlockB(d,
                                      self.dilated_channels,
                                      self.hidden_dim,
                                      self.kernel_size,
                                      self.L) for d in self.dilations])
        self.fc1 = torch.nn.Linear(self.hidden_dim, self.out_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, lst):
        x_ = x
        with torch.no_grad():  # mask does not require gradient
            msk = torch.where(x == PADDING_ITEM, 1, 0).bool()  # compute tensor representation of mask
        x = self.emb_layer(x)
        for dcnn in self.dilated_convolutions:  # dilated convolutional layers
            x = dcnn(x)  # dilated convolution updates previous value
            x[msk] = self.emb_layer(x_[msk])  # re-embed padded items
        x = x[torch.arange(x.size(0)), lst - 1]  # extract last element tensor entry
        x = self.fc1(x)  # fully-connected output layer
        return self.softmax(x)



class GRU4Rec(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        # VARIABLE INITIALIZATIONS
        self.hidden_dim = args.hidden_dim
        self.out_size = args.out_size

        # MODEL LAYERS
        self.emb_layer = torch.nn.Embedding(num_embeddings=self.out_size, embedding_dim=self.hidden_dim, max_norm=None,
                                            padding_idx=PADDING_ITEM)

        # GRU is shape preservint (input_size == hidden_size) -> stackable
        self.gru = torch.nn.GRU(input_size=self.hidden_dim, hidden_size=args.hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = torch.nn.Linear(self.hidden_dim, self.out_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, lst):
        x = self.emb_layer(x)
        # uses lst to determine the depth of the recurrent grus for each input
        # pack sequence x for GRU cell according to the sequence lengths in lst, enforce_sorted=False (onnx not needed)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lst, batch_first=True, enforce_sorted=False)
        _, x = self.gru(x)
        x = x[-1]  # extract last hidden state from output
        x = self.fc1(x)
        return self.softmax(x)


class SAS4Rec(torch.nn.Module):

    def __init__(self, args):
        super(SAS4Rec, self).__init__()

        # VARIABLE INITIALIZATIONS
        self.hidden_dim = args.hidden_dim
        self.out_size = args.out_size
        self.L = args.max_seq_len
        self.emb_ids = torch.tensor(range(self.L)).to('cuda' if args.cuda else 'cpu')
        self.num_attn_blocks = args.sas4rec_num_sablocks
        self.dropout = args.dropout
        self.pw_ff_dim = args.sas4rec_positionwise_feedforward_dim

        # MODEL LAYERS
        self.emb_layer = torch.nn.Embedding(num_embeddings=self.out_size, embedding_dim=self.hidden_dim, max_norm=None,
                                            padding_idx=PADDING_ITEM)
        self.pos_emb = torch.nn.Embedding(num_embeddings=self.L, embedding_dim=self.hidden_dim, max_norm=None)
        self.sa_blocks = torch.nn.ModuleList([SelfAttentionBlock(self.L,
                                                                 self.hidden_dim,
                                                                 self.dropout,
                                                                 self.pw_ff_dim) for _ in range(self.num_attn_blocks)])
        self.fc1 = torch.nn.Linear(self.hidden_dim, self.out_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def _embed(self, x):
        """embeds the items and adds learnable positional embeddings"""
        return self.emb_layer(x) + self.pos_emb(self.emb_ids)

    def forward(self, x, lst):
        x_ = x
        with torch.no_grad():  # mask does not require gradient
            msk = torch.where(x == PADDING_ITEM, 1, 0).bool()  # compute tensor representation of mask
        x = self._embed(x)  # embed items
        for sa_block in self.sa_blocks:
            x = sa_block(x, x, x)
            x[msk] = self.emb_layer(x_[msk])  # re-embed padded items
        x = x[torch.arange(x.size(0)), lst - 1]  # extract last element tensor entry
        x = self.fc1(x)
        return self.softmax(x)


class Random4Rec(torch.nn.Module):  # extends torch.nn.Module to be consistent with other model classes

    def __init__(self, args):
        super().__init__()
        self.num_items = args.out_size
        self.dummy = torch.nn.Linear(1, 1)  # needed to pass empty parameter list check in optimizer

    def forward(self, x):
        """returns a softmax-like probability distributions where one item randomly has prob. 1 and all others 0"""
        t = torch.zeros(size=(x.size(0), self.num_items), requires_grad=False)
        it = torch.randint(low=1, high=self.num_items, size=(x.size(0),))  # generate random indexes
        t[torch.arange(x.size(0)), tuple(it)] = 1  # generate faux-softmax
        return t
