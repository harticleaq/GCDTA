import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from utils.util import get_active_func

class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        _, attn_output_weights = self.multihead_attn(tgt, memory, memory,
                                                               attn_mask=memory_mask,
                                                               key_padding_mask=memory_key_padding_mask)
        return tgt, attn_output_weights

class CustomTransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        layers = [CustomTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        self.decoder = nn.ModuleList(layers)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        for layer in self.decoder:
            tgt, weights = layer(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask)
        
        return tgt, weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(1, 0).to(device)
 
    def forward(self, x):
        return self.pe[:x.size(0), :]
 
class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, d_model, device, nhead,
                 num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerTimeSeriesModel, self).__init__()
 
        self.device = device
        self.positional_encoding = PositionalEncoding(d_model, device)
        # self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
        #                                   num_encoder_layers=num_encoder_layers,
        #                                   num_decoder_layers=num_decoder_layers,
        #                                   dim_feedforward=dim_feedforward,
        #                                   dropout=dropout)
        # decoder_layer = CustomTransformerDecoderLayer(d_model=d_model, 
        #                                            nhead=nhead,
        #                                            dropout=dropout ,
        #                                            dim_feedforward=dim_feedforward)
        self.transformer = CustomTransformerDecoder(num_decoder_layers, d_model,
                                                       nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)
 
    def forward(self, src, tgt, tgt_mask=True, use_pe=True):
        
        if tgt_mask:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(self.device)
            # src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(0)).to(self.device)
        if use_pe:
            src = src + self.positional_encoding(src)
            tgt = tgt + self.positional_encoding(tgt)

        output, att = self.transformer(tgt, src, tgt_mask=tgt_mask)
        # print(att.max(dim=1), att.argmax(dim=1))
        output = self.fc_out(output)[-1]
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args["device"]
        self.att_dim = args["att_dim"]
        self.n_layers = args["n_layers"]
        self.n_heads = args["n_heads"]
        self.smi_hidden_size = args["smi_hidden_size"]
        self.seq_hidden_size = args["seq_hidden_size"]

        self.smi_lr = nn.Linear(self.smi_hidden_size, self.att_dim)
        self.seq_lr = nn.Linear(self.seq_hidden_size, self.att_dim)
        self.cross_attention = TransformerTimeSeriesModel(
                                self.att_dim, 
                                self.device,
                                self.n_heads,
                                self.n_layers,
                                self.n_layers,
                                self.att_dim,
                                dropout=args["dropout"]
                                                        )

    def forward(self, smi, seq):
        smi = self.smi_lr(smi)
        seq = self.seq_lr(seq)
        smi = smi.transpose(0, 1)
        seq = seq.transpose(0, 1)
        smi = self.cross_attention(smi, seq)
        return smi




class CrossAttentionLayer(nn.Module):
    def __init__(self, smi_dim, seq_dim, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(smi_dim, hid_dim)
        self.w_k = nn.Linear(seq_dim, hid_dim)
        self.w_v = nn.Linear(seq_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        #[batch size, sent len, hid dim]
        x = x.permute(0, 2, 1) #[batch size, hid dim, sent len]
        x = self.do(F.relu(self.fc_1(x)))# [batch size, pf dim, sent len]
        x = self.fc_2(x) # [batch size, hid dim, sent len]
        x = x.permute(0, 2, 1) #[batch size, sent len, hid dim]
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, hid_dim, smi_dim, seq_dim, n_heads, dropout=0.2, device="cuda", first_norm=False):
        super().__init__()
        self.first_norm = first_norm
        hid_dim = smi_dim
        self.ln = nn.LayerNorm(hid_dim)
        self.pf = PositionwiseFeedforward(hid_dim, seq_dim, dropout)
        self.sa = CrossAttentionLayer(smi_dim, smi_dim, hid_dim, n_heads, dropout, device)
        self.ea = CrossAttentionLayer(smi_dim, seq_dim, hid_dim, n_heads, dropout, device)
        
        self.ff = nn.Sequential(
            *[
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        
        self.do = nn.Dropout(dropout)

    def forward(self, smi, seq, smi_mask=None, seq_mask=None):
        # smi = [batch_size, smi len, hid_dim]
        # seq = [batch_size, seq len, hid_dim] # encoder output
        # smi_mask = [batch size, smi sent len]
        # seq_mask = [batch size, seq len]
        if self.first_norm:
            smi = smi + self.do(self.pf(self.ln(smi)))
            x = self.ln(smi)
            smi = smi + self.do(self.sa(x, x, x, smi_mask))
            smi = smi + self.do(self.ea(self.ln(smi), seq, seq, seq_mask))
        
        else:
            smi = self.ln(smi + self.do(self.pf(smi)))
            smi = self.ln(smi + self.do(self.sa(smi, smi, smi, smi_mask)))
            smi = self.ln(smi + self.do(self.ea(smi, seq, seq, seq_mask)))
            smi = self.ln(smi + self.ff(smi))
        return smi


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args["device"]
        self.att_dim = args["att_dim"]
        self.n_layers = args["n_layers"]
        self.n_heads = args["n_heads"]
        self.smi_hidden_size = args["smi_hidden_size"]
        self.seq_hidden_size = args["seq_hidden_size"]

        self.cross_attention = nn.ModuleList(
            [
                CrossAttention(self.att_dim, self.smi_hidden_size, self.seq_hidden_size, self.n_heads, first_norm=self.args["first_norm"])
                for _ in range(self.n_layers)
            ]
        )
        self.fc1 = nn.Linear(self.smi_hidden_size, self.smi_hidden_size)
        
        if args["loss_function"] == "mse":
            final_dim = 1 
        elif args["loss_function"] == "ce":
            final_dim = 2
        self.fc2 = nn.Linear(self.smi_hidden_size, final_dim)
        self.final_activation = get_active_func(self.args["final_activation_func"])
    
    def forward(self, smi, seq):
        for layer in self.cross_attention:
            smi = layer(smi, seq)

        # fusion = smi.sum(dim=1)
        fusion = smi[:, -1]
        label = self.final_activation(self.fc1(fusion))
        label = self.fc2(label)
        # label = self.predict_layer(label)
        return label