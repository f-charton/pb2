"""
you give this script some sequences of tokens of the form
V2,V13,V21,V7,V21,V10,V2,V3,V4,V2,V3,V1,V18,V8,V12,V6
(one per line)
and it will generate more things like it.

This is a very mild adaption of Kaparthy's "makemore"
implementation of a baby transformer.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

from logging import getLogger

logger=getLogger()


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, past_kv=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present_kv = (k, v)

        if past_kv is None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else:
            causal_mask = (1.0 - self.bias[:, :, k.size(-2) - q.size(-2) : k.size(-2), : k.size(-2)]).to(q.device)
            causal_mask = causal_mask * torch.finfo(q.dtype).min
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y, present_kv


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class Transformer(nn.Module):
    def __init__(self, config, pad_token_id, eos_token_id):
        super().__init__()
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M")
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kv=None):
        device = idx.device
        b, t = idx.size()

        past_len = 0
        if past_kv is not None and len(past_kv) > 0 and past_kv[0] is not None:
            past_len = past_kv[0][0].size(2)

        pos = torch.arange(past_len, past_len + t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        presents_kv = []
        for i, block in enumerate(self.transformer.h):
            pkv = None if past_kv is None else past_kv[i]
            x, present_kv = block(x, past_kv=pkv)
            presents_kv.append(present_kv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.pad_token_id)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, presents_kv

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        self.eval()

        past_kv = None
        for i in range(max_new_tokens):
            last_tokens = idx[:, -1].unsqueeze(-1)
            finished_mask = torch.zeros_like(last_tokens, dtype=torch.bool) | (last_tokens == self.eos_token_id) | (last_tokens == self.pad_token_id)

            if torch.all(finished_mask):
                idx_next = torch.full_like(finished_mask, self.pad_token_id, dtype=torch.long)
            else:
                if i == 0 or past_kv is None:
                    idx_cond = idx
                else:
                    idx_cond = idx[:, -1].unsqueeze(1)
                logits, _, past_kv = self(idx_cond, past_kv=past_kv)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                if do_sample:
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    _, idx_next = torch.topk(probs, k=1, dim=-1)
                idx_next = torch.where(finished_mask, self.pad_token_id, idx_next)

            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx

    
@torch.inference_mode()
def evaluate(model, dataset, device, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0, collate_fn=dataset.collate_fn)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        _, loss, _ = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss
