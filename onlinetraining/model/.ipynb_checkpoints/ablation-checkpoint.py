from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class PerceiverIO_remove_first_cross(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        seq_dropout_prob = 0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob
        self.num_latents = num_latents
        # self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # self.cross_attend_blocks = nn.ModuleList([
        #     PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
        #     PreNorm(latent_dim, FeedForward(latent_dim))
        # ])
        self.pool_proj = nn.Sequential(
                            nn.Linear(dim, latent_dim * 2),
                            GEGLU()
                        )



        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        queries = None
    ):
        b, n, d = data.shape
        
        # ===== pooling over input sequence =====
        if exists(mask):
            mask = mask.unsqueeze(-1)                    # (b, n, 1)
            data = data * mask
            pooled = data.sum(dim=1) / mask.sum(dim=1).clamp(min=1.)
        else:
            pooled = data.mean(dim=1)                    # (b, dim)
        
        # ===== project to latent_dim =====
        pooled = self.pool_proj(pooled)                  # (b, latent_dim)
        
        # ===== broadcast to latents =====
        x = repeat(pooled, 'b d -> b n d', n=self.num_latents)

        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if not exists(queries):
            return x

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents
        
        latents = self.decoder_cross_attn(queries, context = x)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out

        return self.to_logits(latents)

class PerceiverIO_remove_all_cross(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        seq_dropout_prob = 0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob
        self.num_latents = num_latents

        # encoder: pooling(data) -> latent_dim
        self.pool_proj = nn.Sequential(
            nn.Linear(dim, latent_dim * 2),
            GEGLU()
        )

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
        get_latent_ff   = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_pool = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            GEGLU()
        ) if decoder_ff else nn.LayerNorm(latent_dim)

        # 现在 decoder 输出来自 latent_dim，因此 to_logits 应该吃 latent_dim
        self.to_logits = nn.Linear(latent_dim, logits_dim) if exists(logits_dim) else nn.Identity()


    def forward(self, data, mask=None, queries=None):
        b, n, d = data.shape

        # ---- encoder pooling over input ----
        if exists(mask):
            m = mask.unsqueeze(-1)                  # (b, n, 1)
            data_masked = data * m
            pooled = data_masked.sum(dim=1) / m.sum(dim=1).clamp(min=1.)
        else:
            pooled = data.mean(dim=1)               # (b, dim)

        pooled = self.pool_proj(pooled)             # (b, latent_dim)
        x = repeat(pooled, 'b d -> b n d', n=self.num_latents)  # (b, num_latents, latent_dim)

        # ---- latent self-attn blocks (保留！) ----
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # ---- decoder: pool latents to a single vector ----
        # 最简单：mean over latent slots
        dec = x.mean(dim=1)                          # (b, latent_dim)
        dec = self.decoder_pool(dec)                 # (b, latent_dim)  (或 LayerNorm)

        # ---- match output shape to queries (if provided) ----
        if exists(queries):
            if queries.ndim == 2:
                # (nq, qdim) -> (b, nq, qdim)
                queries = repeat(queries, 'n d -> b n d', b=b)
            nq = queries.shape[1]
            dec = repeat(dec, 'b d -> b n d', n=nq)   # (b, nq, latent_dim)
            return self.to_logits(dec)                # (b, nq, logits_dim)

        # 如果没给 queries，就返回一个全局输出（你下游自己决定怎么用）
        return self.to_logits(dec)                    # (b, logits_dim) 或 (b, latent_dim) 取决于 logits_dim
        
class LatentMixer(nn.Module):
    def __init__(self, num_latents, dim):
        super().__init__()

        self.token_mlp = nn.Sequential(
            nn.LayerNorm(num_latents),
            nn.Linear(num_latents, num_latents),
            nn.GELU()
        )

        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            GEGLU()
        )

    def forward(self, x):

        # token mixing
        x = x + self.token_mlp(x.transpose(1, 2)).transpose(1, 2)

        # channel mixing
        x = x + self.channel_mlp(x)

        return x


class PerceiverIO_no_attn(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        seq_dropout_prob = 0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob
        self.num_latents = num_latents

        # encoder: pooling(data) -> latent_dim
        self.pool_proj = nn.Sequential(
            nn.Linear(dim, latent_dim * 2),
            GEGLU()
        )


        self.layers = nn.ModuleList([
            LatentMixer(num_latents, latent_dim)
            for _ in range(depth)
        ])

        self.decoder_pool = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            GEGLU()
        ) if decoder_ff else nn.LayerNorm(latent_dim)

        # 现在 decoder 输出来自 latent_dim，因此 to_logits 应该吃 latent_dim
        self.to_logits = nn.Linear(latent_dim, logits_dim) if exists(logits_dim) else nn.Identity()


    def forward(self, data, mask=None, queries=None):
        b, n, d = data.shape

        # ---- encoder pooling over input ----
        if exists(mask):
            m = mask.unsqueeze(-1)                  # (b, n, 1)
            data_masked = data * m
            pooled = data_masked.sum(dim=1) / m.sum(dim=1).clamp(min=1.)
        else:
            pooled = data.mean(dim=1)               # (b, dim)

        pooled = self.pool_proj(pooled)             # (b, latent_dim)
        x = repeat(pooled, 'b d -> b n d', n=self.num_latents)  # (b, num_latents, latent_dim)


        for mixer in self.layers:
            x = mixer(x)
        # ---- decoder: pool latents to a single vector ----
        # 最简单：mean over latent slots
        dec = x.mean(dim=1)                          # (b, latent_dim)
        dec = self.decoder_pool(dec)                 # (b, latent_dim)  (或 LayerNorm)

        # ---- match output shape to queries (if provided) ----
        if exists(queries):
            if queries.ndim == 2:
                # (nq, qdim) -> (b, nq, qdim)
                queries = repeat(queries, 'n d -> b n d', b=b)
            nq = queries.shape[1]
            dec = repeat(dec, 'b d -> b n d', n=nq)   # (b, nq, latent_dim)
            return self.to_logits(dec)                # (b, nq, logits_dim)

        # 如果没给 queries，就返回一个全局输出（你下游自己决定怎么用）
        return self.to_logits(dec)                    # (b, logits_dim) 或 (b, latent_dim) 取决于 logits_dim
        



class CVA(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        seq_dropout_prob = 0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob
        self.num_latents = num_latents

        # encoder: pooling(data) -> latent_dim
        self.pool_proj = nn.Sequential(
            nn.Linear(dim, latent_dim * 2),
            GEGLU()
        )

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
        get_latent_ff   = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_pool = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            GEGLU()
        ) if decoder_ff else nn.LayerNorm(latent_dim)

        # 现在 decoder 输出来自 latent_dim，因此 to_logits 应该吃 latent_dim
        self.to_logits = nn.Linear(latent_dim, logits_dim) if exists(logits_dim) else nn.Identity()


    def forward(self, data, mask=None, queries=None):
        b, n, d = data.shape

        # ---- encoder pooling over input ----
        if exists(mask):
            m = mask.unsqueeze(-1)                  # (b, n, 1)
            data_masked = data * m
            pooled = data_masked.sum(dim=1) / m.sum(dim=1).clamp(min=1.)
        else:
            pooled = data.mean(dim=1)               # (b, dim)

        pooled = self.pool_proj(pooled)             # (b, latent_dim)
        x = repeat(pooled, 'b d -> b n d', n=self.num_latents)  # (b, num_latents, latent_dim)

        # ---- latent self-attn blocks (保留！) ----
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # ---- decoder: pool latents to a single vector ----
        # 最简单：mean over latent slots
        dec = x.mean(dim=1)                          # (b, latent_dim)
        dec = self.decoder_pool(dec)                 # (b, latent_dim)  (或 LayerNorm)

        # ---- match output shape to queries (if provided) ----
        if exists(queries):
            if queries.ndim == 2:
                # (nq, qdim) -> (b, nq, qdim)
                queries = repeat(queries, 'n d -> b n d', b=b)
            nq = queries.shape[1]
            dec = repeat(dec, 'b d -> b n d', n=nq)   # (b, nq, latent_dim)
            return self.to_logits(dec)                # (b, nq, logits_dim)

        return self.to_logits(dec)                  


class MLP(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, queries=None):
        x = x.mean(dim=1) 
        return self.net(x)

class PoolingIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, queries=None):
        x = x.mean(dim=1) 
        return x

