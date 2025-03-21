from common import dim, seq_len, dropout, vocab_size, policy_dim, dim_head, num_heads
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self):
        super().__init__()
        indices = torch.arange(dim_head // 2)
        indices = -2 * (indices - 1) / dim_head
        thetas = 10_000**indices
        m = torch.arange(seq_len).view(-1, 1)
        vals = thetas * m  # (seq_len, d // 2)
        self.register_buffer("cosines", torch.cos(vals))
        self.register_buffer("sines", torch.sin(vals))

    def forward(self, x):
        x_seq_len = x.shape[-2]
        output = torch.zeros_like(x)
        output[..., ::2] = (
            self.cosines[:x_seq_len] * x[..., ::2]
            - self.sines[:x_seq_len] * x[..., 1::2]
        )
        output[..., 1::2] = (
            self.sines[:x_seq_len] * x[..., ::2]
            + self.cosines[:x_seq_len] * x[..., 1::2]
        )
        return output


class RoPESingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = RoPE()
        return cls._instance


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.rope = RoPESingleton.get_instance()
        self.keys_cache = None
        self.values_cache = None

    def reset_cache(self):
        self.keys_cache = None
        self.values_cache = None

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(
            3, dim=-1
        )  # shape is now (batch_size, seq_len, dim_head * heads) for q,k,v

        q = q.view(batch_size, seq_len, num_heads, dim_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, dim_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, dim_head).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        if self.training:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout
            )
        else:
            # cat along the seq dimension
            self.keys_cache = (
                torch.cat([self.keys_cache, k], -2)
                if self.keys_cache is not None
                else k
            )
            self.values_cache = (
                torch.cat([self.values_cache, v], -2)
                if self.values_cache is not None
                else v
            )
            out = F.scaled_dot_product_attention(
                q, self.keys_cache, self.values_cache, dropout_p=0.0, is_causal=False
            )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.to_out(out)


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = FFN(dim)  # MoE(dim, num_experts=4) #FFN(dim)
        self.attn = Attention(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.num_program_tokens = 200
        self.embedding_table = nn.Embedding(vocab_size, dim)
        self.layers = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_layers)])
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)
        # self.program_head = nn.Linear(dim, self.num_program_tokens)
        self.to_logits.weight = self.embedding_table.weight
        self.norm = nn.LayerNorm(dim)

        # self.value_head = nn.Sequential(
        #     *[TransformerBlock(dim) for _ in range(2)],
        #     nn.Linear(dim, 1),
        #     nn.Tanh(),
        # )

        # self.policy_head = nn.Sequential(
        #     *[TransformerBlock(dim) for _ in range(2)],
        #     nn.Linear(dim, policy_dim),
        #     nn.Softmax(dim=-1),
        # )

    def forward(self, x, target=None, rl=False, output_programs=False, temperature=1.0):
        B, T = x.shape
        self.pos_range = torch.arange(T, device=x.device)
        x = self.embedding_table(x)
        x = self.norm(self.layers(x))

        # if rl:
        #     value = self.value_head(x)
        #     policy = self.policy_head(x)
        #     return policy, value

        # if output_programs:
        #     vocab_logits = self.to_logits(x)  # (B, T, vocab_size)
        #     program_logits = self.program_head(x)  # (B, T, num_program_tokens)
        #     x = torch.cat([vocab_logits, program_logits], dim=-1)
        # else:
        # we just do a normal projection for pretraining
        x = self.to_logits(x)  # shape is (batch_size, seq_len, vocab_size)

        if target is not None:
            x = x.view(B * T, vocab_size)
            y = target.view(B * T)

            loss = F.cross_entropy(x, y)
            return loss

        priors = F.softmax(x[:, -1, :] / temperature, dim=-1)
        sample = torch.multinomial(priors, num_samples=1)
        return priors, sample
