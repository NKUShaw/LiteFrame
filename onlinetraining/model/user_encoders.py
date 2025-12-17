# -*- coding: utf-8 -*-

import math
import torch
import torch.fft
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import uniform_, xavier_normal_, constant_, xavier_uniform_

from .modules import TransformerEncoder

class User_Encoder_SASRec(torch.nn.Module):
    def __init__(self, args):
        super(User_Encoder_SASRec, self).__init__()

        self.transformer_encoder = TransformerEncoder(n_vocab=None, n_position=args.max_seq_len,
                                                      d_model=args.embedding_dim, n_heads=args.num_attention_heads,
                                                      dropout=args.drop_rate, n_layers=args.transformer_block)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank):
        att_mask = (log_mask != 0)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0., -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)
        
class User_Encoder_NextItNet(nn.Module):
    def __init__(self, args): 
        super(User_Encoder_NextItNet, self).__init__()

        self.residual_channels = args.embedding_dim 
        self.block_num = args.block_num
        self.dilations = [1, 4] * self.block_num 
        self.kernel_size = 3 

        # residual blocks    
        # dilations in blocks:[1,2,4,8,1,2,4,8,...] for a, [1,4,1,4,1,4,...] for b
        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)
        self.final_layer = nn.Linear(self.residual_channels, self.residual_channels)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / (self.output_dim+1))
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, item_seq_emb): #, pos, neg
        return self.residual_blocks(item_seq_emb)

    def predict(self, item_seq_emb):
        dilate_outputs = self.residual_blocks(item_seq_emb)
        seq_output = self.final_layer(dilate_outputs)  # [batch_size, seq_len, embedding_size]
        return seq_output  

class ResidualBlock_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)
        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 + x

    def conv_pad(self, x, dilation):
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad


class User_Encoder_GRU4Rec(nn.Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:
        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, args):
        super().__init__()

        self.embedding_size = args.embedding_dim
        self.n_layers = args.block_num
        self.hidden_size = args.embedding_dim
        self.dropout = args.drop_rate

        # define layers
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.emb_dropout = nn.Dropout(self.dropout)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq_emb):
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)

        return gru_output

class User_Encoder_NARM(nn.Module):
    def __init__(self, args):
        super(User_Encoder_NARM, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_size = args.embedding_dim
        self.n_layers = args.block_num
        
        # 1. GRU 
        self.gru = nn.GRU(
            self.embedding_dim,
            self.hidden_size,
            self.n_layers,
            batch_first=True
        )
        
        # 2. Attention 
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, item_seq_emb, log_mask=None, local_rank=None):
        # item_seq_emb: [B, L, D]
        
        # GRU 输出
        # outputs: [B, L, H], hidden: [n_layers, B, H]
        gru_out, hidden = self.gru(item_seq_emb)
        
        c_global = hidden[-1] # [B, H]
        
        # --- Local Encoder (Attention) ---
        # q1: [B, L, H], q2: [B, H] -> [B, 1, H]
        q1 = self.a_1(gru_out)
        q2 = self.a_2(c_global).unsqueeze(1)

        alpha = self.v_t(torch.sigmoid(q1 + q2)) # [B, L, 1]
        
        if log_mask is not None:
             mask = (log_mask > 0).unsqueeze(-1) # [B, L, 1]
             alpha = alpha.masked_fill(~mask, -float("inf"))
             
        weights = torch.softmax(alpha, dim=1) # [B, L, 1]
        
        # 加权求和得到 Local Context
        c_local = torch.sum(gru_out * weights, dim=1) # [B, H]
        final_rep = gru_out + c_local.unsqueeze(1)
        
        return final_rep


class User_Encoder_FMLPRec(nn.Module):
    r"""
    FMLP-Rec: Filter-enhanced MLP for Sequential Recommendation.
    Uses FFT to mix information globally in the frequency domain, replacing Self-Attention.
    """
    def __init__(self, args):
        super(User_Encoder_FMLPRec, self).__init__()
        self.n_layers = args.block_num
        self.embedding_dim = args.embedding_dim
        self.max_seq_len = args.max_seq_len
        self.dropout_rate = args.drop_rate

        # Stack FMLP Blocks
        self.blocks = nn.ModuleList([
            FMLP_Block(args) for _ in range(self.n_layers)
        ])
        
        self.LayerNorm = nn.LayerNorm(self.embedding_dim, eps=1e-12)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, item_seq_emb, log_mask=None, local_rank=None):
        # item_seq_emb: [B, L, D]
        # log_mask: [B, L] (Optional)
        
        x = item_seq_emb
        
        # FMLP blocks (FFT based)
        for block in self.blocks:
            x = block(x)
            
        x = self.LayerNorm(x)

        # Apply mask if provided (to zero out padding in the output)
        if log_mask is not None:
            mask = (log_mask != 0).unsqueeze(-1).float() # [B, L, 1]
            x = x * mask
            
        return x

class FMLP_Block(nn.Module):
    def __init__(self, args):
        super(FMLP_Block, self).__init__()
        self.filter_layer = FilterLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, x):
        x = self.filter_layer(x)
        x = self.feed_forward(x)
        return x

class FilterLayer(nn.Module):
    def __init__(self, args):
        super(FilterLayer, self).__init__()
        self.max_seq_len = args.max_seq_len
        self.embedding_dim = args.embedding_dim
        
        # Learnable Complex Weights for Frequency Domain
        self.complex_weight = nn.Parameter(
            torch.randn(1, args.max_seq_len // 2 + 1, args.embedding_dim, 2, dtype=torch.float32) * 0.02
        )
        
        self.out_dropout = nn.Dropout(args.drop_rate)
        self.LayerNorm = nn.LayerNorm(args.embedding_dim, eps=1e-12)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        
        residual = x
        
        with torch.cuda.amp.autocast(enabled=False):
            x_f32 = x.to(torch.float32)
            
            # 1. FFT
            x_fft = torch.fft.rfft(x_f32, dim=1, norm='ortho')
            
            # 2. Filtering
            weight = torch.view_as_complex(self.complex_weight)
            
            # 动态适配长度
            current_fft_len = x_fft.shape[1]
            if current_fft_len != weight.shape[1]:
                if current_fft_len < weight.shape[1]:
                    weight = weight[:, :current_fft_len, :]
                else:
                    pad = torch.zeros(1, current_fft_len - weight.shape[1], D, device=x.device, dtype=weight.dtype)
                    weight = torch.cat([weight, pad], dim=1)
            x_fft = x_fft * weight
            
            # 3. IFFT
            x_out = torch.fft.irfft(x_fft, n=L, dim=1, norm='ortho')
    
        x = x_out.to(residual.dtype)
        
        # 4. Residual & Norm
        x = self.out_dropout(x)
        x = self.LayerNorm(x + residual)
        
        return x

class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(args.embedding_dim, args.embedding_dim * 4)
        self.dense_2 = nn.Linear(args.embedding_dim * 4, args.embedding_dim)
        self.LayerNorm = nn.LayerNorm(args.embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(args.drop_rate)
        
    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        residual = x
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        x = self.LayerNorm(x + residual)
        return x

class User_Encoder_Bert4Rec(torch.nn.Module):
    r"""
    BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer.
    """
    def __init__(self, args):
        super(User_Encoder_Bert4Rec, self).__init__()

        self.transformer_encoder = TransformerEncoder(n_vocab=None, n_position=args.max_seq_len,
                                                      d_model=args.embedding_dim, n_heads=args.num_attention_heads,
                                                      dropout=args.drop_rate, n_layers=args.transformer_block)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank):
        # input_embs: Item embeddings [B, L, D]
        # log_mask: Sequence padding mask [B, L]
        
        # 1. (Padding Mask)
        att_mask = (log_mask != 0) # 
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        
        # 2. BERT4Rec Bidirectional Mask
        att_mask = att_mask.expand((-1, -1, log_mask.size(-1), -1)).to(local_rank)
        
        # 将 True/False 掩码转换为 Transformer Attention 所需的浮点掩码：
        # 0. 表示有效 (Attention Score * 0)，-1e9 表示无效 (Attention Score * (-inf))
        att_mask = torch.where(att_mask, 0., -1e9) 
        
        # 3. 传入 Transformer Encoder
        # TransformerEncoder 会返回序列中每个位置的学习到的双向表示 [B, L, D]
        return self.transformer_encoder(input_embs, log_mask, att_mask)
