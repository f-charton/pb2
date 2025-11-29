"""
you give this script some sequences of tokens of the form
V2,V13,V21,V7,V21,V10,V2,V3,V4,V2,V3,V1,V18,V8,V12,V6
(one per line)
and it will generate more things like it.

This is a very mild adaption of Kaparthy's "makemore"
implementation of a baby transformer.
"""

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List
import numpy as np
import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

from logging import getLogger

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

logger=getLogger()
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_head: int = 4

# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, past_kv=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present_kv = (k, v)

        if past_kv is None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else:
            causal_mask = (1.0 - self.bias[:, :, k.size(-2) - q.size(-2) : k.size(-2), :k.size(-2)]).to(q.device)
            causal_mask = causal_mask * torch.finfo(q.dtype).min
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y, present_kv

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlpf(self.ln_2(x))
        return x, present_kv

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None, past_kv=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # figure out positional offset if we're continuing generation
        past_len = 0
        if past_kv is not None and len(past_kv) > 0 and past_kv[0] is not None:
            past_len = past_kv[0][0].size(2)

        pos = torch.arange(past_len, past_len + t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb

        presents_kv = []
        for i, block in enumerate(self.transformer.h):
            pkv = None if past_kv is None else past_kv[i]
            x, present_kv = block(x, past_kv=pkv)
            presents_kv.append(present_kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss, presents_kv

# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    model.eval()
    block_size = model.get_block_size()
    B = idx.size(0)

    past_kv = None

    for i in range(max_new_tokens):
        if i == 0 or past_kv is None:
            idx_cond = idx
        else:
            idx_cond = idx[:, -1].unsqueeze(1)
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx_cond if idx_cond.size(1) <= block_size else idx_cond[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _, presents = model(idx_cond, past_kv=past_kv)
        past_kv = presents
        for n_layer in range(len(past_kv)):
            k_l, v_l = past_kv[n_layer]
            if k_l.size(2) > block_size:
                past_kv[n_layer] = (k_l[:, :, -block_size:, :], v_l[:, :, -block_size:, :])

        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def print_samples(args,model,train_dataset,num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    samples = []
#    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        samples.append(word_samp)
    print('-'*80)
    print(f"{len(samples)} samples:")
    for word in samples:
        print(word)
    print('-'*80)

    
@torch.inference_mode()
def evaluate(model, dataset, device, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss, _ = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

def logprobs(args,model,dataset):
    """Return the log of the probability that the model will generate a given sequence.
    
    Note: What we actually calculate is the probability given a sequence (A,B,..,X) that the
    model will generate a sequence (A,B,...X,...). I.e. we don't care that it stops correctly.
    
    """

    encoded_words = torch.stack(tuple([dataset[i][0] for i in range(len(dataset))])).to(args.device)
    logits, _, _ = model(encoded_words)
    logits = logits.to('cpu')
    probs = F.softmax(logits, dim=-1).detach().numpy() # a tensor of shape (len(dataset), block_size, #tokens+1)

    logprobs_out = []
    for i in range(len(dataset)):
        logprob = 0
        for j in range(dataset.get_output_length()-1):
            if dataset[i][1][j] > 0: ## See remark above: we don't care that it stops correctly.
                logprob += np.log( probs[i,j,dataset[i][1][j]])
        logprobs_out.append(logprob)
    return logprobs_out

# -----------------------------------------------------------------------------
