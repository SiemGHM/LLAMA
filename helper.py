import torch
import torch.nn as nn
import torch.nn.functional as F


class LLAMAConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 512,            # embedding / hidden dimension
        n_layers: int = 8,         # number of transformer blocks
        n_heads: int = 8,          # attention heads
        n_kv_heads: int = 4,       # key/value heads (GQA; set == n_heads to disable)
        max_seq_len: int = 2048,   # max context length
        ffn_dim_multiplier: float = None,  # optional FFN width override
        multiple_of: int = 256,    # FFN dim rounded to this multiple
        norm_eps: float = 1e-5,    # RMSNorm epsilon
        rope_theta: float = 10000.0,  # RoPE base frequency
        dropout: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.dropout = dropout

        # derived
        self.head_dim = dim // n_heads

    

class LLAMATokenizer:
    def __init__(self, bpe_tokenizer, llama_config):
        self.vocab_size = llama_config.vocab_size
        self.tokenizer = bpe_tokenizer
        


def precompute_freqs_cis(head_dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len)
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(q, k, freqs_cis):
    freqs_cis = freqs_cis[:q.shape[2]]
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    q_out = torch.view_as_real(q_ * freqs_cis).flatten(3).type_as(q)
    k_out = torch.view_as_real(k_ * freqs_cis).flatten(3).type_as(k)
    return q_out, k_out


class LLAMAModel(nn.Module):
    def __init__(self, llama_config):
        super().__init__()
        self.config = llama_config
        self.embeddings = self.initialize_embeddings()
        self.transformer_blocks = self.initialize_transformer_blocks()
        self.rms_norm = self.initialize_rms_norm()
        self.linear = nn.Linear(llama_config.dim, llama_config.vocab_size)
        self.register_buffer('freqs_cis', precompute_freqs_cis(llama_config.head_dim, llama_config.max_seq_len, llama_config.rope_theta))
        
    def initialize_embeddings(self):
        return nn.Embedding(self.config.vocab_size, self.config.dim)
    
    def initialize_transformer_blocks(self):
        
        return nn.ModuleList([self.initialize_transformer_block() for _ in range(self.config.n_layers)])
    
    def initialize_transformer_block(self):
        return nn.ModuleDict({
            'rms_norm': RMSNorm(self.config.dim, self.config.norm_eps),
            'multi_head_attention': MultiheadAttention(self.config),
            'ffn': SwiGLU(self.config),
            'rms_norm_2': RMSNorm(self.config.dim, self.config.norm_eps),
        })

    def initialize_rms_norm(self):
        return RMSNorm(self.config.dim, self.config.norm_eps)

    def initialize_mha_attention(self):
        return MultiheadAttention(self.config)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        for block in self.transformer_blocks:
            x = x + block['multi_head_attention'](block['rms_norm'](x), self.freqs_cis)
            x = x + block['ffn'](block['rms_norm_2'](x))
        x = self.rms_norm(x)
        logits = self.linear(x)
        return logits
        
        
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        norm_x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight
    
    
class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.register_buffer('tril', torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).unsqueeze(0).unsqueeze(0))
        
        
    def forward(self, x, freqs_cis):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, freqs_cis)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights.masked_fill(self.tril[:, :, :seq_len, :seq_len] == 0, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        # hidden_dim: 2/3 * 4 * dim, rounded up to nearest multiple_of
        hidden_dim = int(2 * 4 * config.dim / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)

        self.gate_proj = nn.Linear(config.dim, hidden_dim, bias=False)  # W1
        self.up_proj   = nn.Linear(config.dim, hidden_dim, bias=False)  # W3
        self.down_proj = nn.Linear(hidden_dim, config.dim, bias=False)  # W2

    def forward(self, x):
        # SwiGLU: silu(gate) * up, then project down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))