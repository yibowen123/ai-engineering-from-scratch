"""Refined SFT loop: a REAL training step with PyTorch autograd.

Companion to ``main.py``. The original file demonstrates the *shape* of an SFT
loop (instruction formatting, response-only loss masking, the softmax
cross-entropy gradient formula) but then stubs out the actual learning: it
computes ``dlogits`` and throws it away, and "updates" only the FFN weights with
random Gaussian noise. As a result the model never actually learns.

This version keeps the exact same data pipeline and loss-masking idea, but does
the real thing:

  * a small GPT built from ``torch.nn`` (all parameters are learnable),
  * a genuine forward pass,
  * ``loss.backward()`` -> autograd computes gradients for EVERY parameter,
  * ``optimizer.step()`` -> AdamW updates every parameter toward lower loss.

Run with the project venv that has torch installed:
    .venv/bin/python refined_main.py

You should see the training loss fall sharply as the model memorizes the tiny
instruction set -- the thing main.py could not do.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Data: identical instruction set + formatting to main.py (kept self-contained
# so this file is torch-only and does not import the numpy model).
# --------------------------------------------------------------------------
INSTRUCTION_DATA = [
    {"instruction": "What is the capital of France?",
     "response": "The capital of France is Paris."},
    {"instruction": "Explain gravity in one sentence.",
     "response": "Gravity is the force that attracts objects with mass toward each other."},
    {"instruction": "Write a haiku about the ocean.",
     "response": "Waves crash on the shore, salt and foam beneath the sun, endless blue expanse."},
    {"instruction": "What is 15 multiplied by 7?",
     "response": "15 multiplied by 7 is 105."},
    {"instruction": "Name three programming languages.",
     "response": "Three programming languages are Python, Rust, and TypeScript."},
    {"instruction": "Summarize photosynthesis.",
     "response": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen."},
    {"instruction": "What year did World War II end?",
     "response": "World War II ended in 1945."},
    {"instruction": "Define machine learning.",
     "response": "Machine learning is a field where algorithms learn patterns from data to make predictions."},
]

SPECIAL_TOKENS = {"INST_START": 253, "INST_END": 254, "RESP_START": 255}
PAD_ID = 0


def tokenize_instruction_pair(instruction, response, vocab_size=256):
    inst = [min(t, vocab_size - 4) for t in instruction.encode("utf-8")]
    resp = [min(t, vocab_size - 4) for t in response.encode("utf-8")]
    return (
        [SPECIAL_TOKENS["INST_START"]] + inst
        + [SPECIAL_TOKENS["INST_END"], SPECIAL_TOKENS["RESP_START"]] + resp
    )


def create_loss_mask(tokens):
    """1.0 on response tokens only; 0.0 on instruction/special tokens.

    This is the heart of SFT: we optimize the model to produce the response,
    never to reproduce the instruction.
    """
    mask = [0.0] * len(tokens)
    in_response = False
    for i, tok in enumerate(tokens):
        if tok == SPECIAL_TOKENS["RESP_START"]:
            in_response = True
            continue
        if in_response:
            mask[i] = 1.0
    return mask


# --------------------------------------------------------------------------
# Model: a minimal GPT in torch. Every tensor here is an nn.Parameter, so
# autograd tracks it and the optimizer updates it -- unlike main.py, which only
# ever touched the FFN weights (and only with random noise).
# --------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out                 # residual around attention
        x = x + self.ffn(self.ln2(x))    # residual around feed-forward
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=128, num_heads=4,
                 num_layers=4, max_seq_len=128, ff_dim=512):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        _, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos)[None, :, :]
        # Causal mask: position t may attend only to positions <= t.
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=idx.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, causal)
        return self.head(self.ln_f(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# LoRA (Low-Rank Adaptation): instead of updating a big weight matrix W
# (shape out x in), we FREEZE W and learn a low-rank correction dW = (B @ A) *
# scaling, where A is (r x in) and B is (out x r) with r << min(out, in). Only
# A and B are trained, so the number of trainable parameters drops by orders of
# magnitude, yet the effective update dW is full-size. B starts at zero so the
# adapted model is initially identical to the base model (dW = 0).
# --------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base                       # the pretrained/base Linear
        for p in self.base.parameters():
            p.requires_grad = False            # frozen: no gradient, never updated
        self.r = r
        self.scaling = alpha / r               # standard LoRA scaling
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B left at zero => initial dW = B @ A = 0 => starts from the base model.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # base(x) = x W^T + b ;  low-rank delta = (x A^T) B^T * scaling
        delta = (self.dropout(x) @ self.A.t()) @ self.B.t()
        return self.base(x) + delta * self.scaling


def apply_lora(model, r=8, alpha=16, dropout=0.0):
    """Freeze the whole model, then wrap the FFN linears with LoRA adapters.
    Only the injected A/B matrices remain trainable; embeddings, LayerNorms, the
    attention block, the base linear weights and the head all stay frozen.

    Note we target the FFN linears specifically. torch's ``nn.MultiheadAttention``
    runs a fused functional kernel that reads ``out_proj.weight``/``.bias`` as raw
    attributes rather than calling the submodule, so a module wrapper there is
    not picked up. A production LoRA (e.g. PEFT) implements attention manually to
    adapt q/k/v; the FFN linears illustrate the same mechanism cleanly here.
    """
    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks:
        block.ffn[0] = LoRALinear(block.ffn[0], r, alpha, dropout)   # up-projection
        block.ffn[2] = LoRALinear(block.ffn[2], r, alpha, dropout)   # down-projection
    return model


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


# --------------------------------------------------------------------------
# Loss: response-only masked cross-entropy, identical semantics to main.py's
# masked_cross_entropy_loss -- but differentiable end to end via autograd.
# --------------------------------------------------------------------------
def masked_cross_entropy_loss(logits, targets, loss_mask):
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none")
    ce = ce.reshape(B, T) * loss_mask
    denom = loss_mask.sum().clamp(min=1.0)   # average over response tokens only
    return ce.sum() / denom


def build_batch(dataset, seq_len=128):
    """Tokenize + mask every example, then right-pad to a single padded batch.

    Right padding + causal attention means real tokens never attend to pad
    positions, so no separate padding-attention mask is needed. Padded targets
    get loss_mask 0, so they contribute nothing to the loss.
    """
    seqs, masks = [], []
    for ex in dataset:
        toks = tokenize_instruction_pair(ex["instruction"], ex["response"])[:seq_len]
        seqs.append(toks)
        masks.append(create_loss_mask(toks))

    max_len = max(len(s) for s in seqs)
    input_ids, target_ids, loss_mask = [], [], []
    for toks, m in zip(seqs, masks):
        pad = max_len - len(toks)
        padded = toks + [PAD_ID] * pad
        padded_mask = m + [0.0] * pad
        # Next-token teacher forcing: predict token t+1 from tokens <= t.
        input_ids.append(padded[:-1])
        target_ids.append(padded[1:])
        loss_mask.append(padded_mask[1:])   # shift mask to align with targets

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(target_ids, dtype=torch.long),
        torch.tensor(loss_mask, dtype=torch.float32),
    )


def sft_train(model, dataset, num_epochs=60, lr=3e-4, seq_len=128,
              use_lora=False, lora_r=8, lora_alpha=16):
    input_ids, target_ids, loss_mask = build_batch(dataset, seq_len)

    if use_lora:
        model = apply_lora(model, r=lora_r, alpha=lora_alpha)

    # Full fine-tuning optimizes ALL parameters; LoRA optimizes only the small
    # A/B adapters. Either way, autograd + optimizer.step() do the real work.
    params = trainable_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=lr)

    total = model.count_parameters()
    trainable = sum(p.numel() for p in params)
    mode = f"LoRA (r={lora_r}, alpha={lora_alpha})" if use_lora else "Full fine-tuning"
    total_resp = int(loss_mask.sum().item())
    print(f"SFT Training [{mode}]: {len(dataset)} examples, {num_epochs} epochs, lr={lr}")
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable / total:.1%})")
    print(f"Batch tensor: {tuple(input_ids.shape)}  |  response tokens/step: {total_resp}")
    print()

    losses = []
    model.train()
    for epoch in range(num_epochs):
        logits = model(input_ids)                              # forward
        loss = masked_cross_entropy_loss(logits, target_ids, loss_mask)

        optimizer.zero_grad()                                  # clear old grads
        loss.backward()                                        # autograd: dL/dparam for EVERY param
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilize
        optimizer.step()                                       # param -= lr * update(grad)

        losses.append(loss.item())
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1:3d}/{num_epochs} | Loss: {loss.item():.4f}")

    return model, losses


@torch.no_grad()
def generate_response(model, prompt_tokens, max_new_tokens=40, temperature=0.7):
    model.eval()
    tokens = list(prompt_tokens)
    for _ in range(max_new_tokens):
        ctx = torch.tensor(tokens[-model.max_seq_len:], dtype=torch.long)[None, :]
        logits = model(ctx)[0, -1, :]
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1).item())
        tokens.append(nxt)
        if nxt == SPECIAL_TOKENS["INST_START"]:   # stop if it starts a new turn
            break
    return tokens


@torch.no_grad()
def measure_forgetting(model, test_text, seq_len=64):
    """Mean next-token loss on held-out raw text (unmasked), like main.py."""
    model.eval()
    data = [min(b, 252) for b in test_text.encode("utf-8")[:512]]
    toks = torch.tensor(data, dtype=torch.long)
    total, n = 0.0, 0
    for start in range(0, len(toks) - seq_len - 1, seq_len):
        inp = toks[start:start + seq_len][None, :]
        tgt = toks[start + 1:start + seq_len + 1][None, :]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        total += loss.item()
        n += 1
    return total / max(n, 1)


def evaluate_instruction_following(model, instructions):
    print("Evaluating instruction following:")
    print("-" * 50)
    for instruction in instructions:
        prompt = (
            [SPECIAL_TOKENS["INST_START"]]
            + [min(t, 252) for t in instruction.encode("utf-8")]
            + [SPECIAL_TOKENS["INST_END"], SPECIAL_TOKENS["RESP_START"]]
        )
        out = generate_response(model, prompt, max_new_tokens=40, temperature=0.6)
        resp = out[len(prompt):]
        text = bytes([t for t in resp if t < 128]).decode("utf-8", errors="replace")
        print(f"  Q: {instruction}")
        print(f"  A: {text[:80]}")
        print()


if __name__ == "__main__":
    import sys

    # Toggle LoRA with:  python refined_main.py --lora
    use_lora = "--lora" in sys.argv
    # LoRA has far fewer trainable params, so it tolerates (and benefits from) a
    # higher learning rate than full fine-tuning.
    lr = 1e-3 if use_lora else 3e-4

    torch.manual_seed(42)

    test_text = (
        "The transformer architecture processes sequences through self-attention.\n"
        "Each layer applies multi-head attention followed by a feedforward network.\n"
        "Residual connections and layer normalization stabilize deep networks.\n"
        "The model learns to predict the next token given all previous tokens."
    )

    print("=" * 70)
    print("INSTRUCTION TUNING (SFT) DEMO — real autograd / optimizer.step")
    print(f"Mode: {'LoRA' if use_lora else 'Full fine-tuning'}")
    print("=" * 70)
    print()

    model = MiniGPT(vocab_size=256, embed_dim=128, num_heads=4,
                    num_layers=4, max_seq_len=128, ff_dim=512)
    print(f"Model: {model.count_parameters():,} parameters")
    print()

    print("PRE-SFT: base model loss on raw text")
    base_loss = measure_forgetting(model, test_text)
    print(f"  Base model loss: {base_loss:.4f}\n")

    print("=" * 70)
    print("SFT TRAINING")
    print("=" * 70)
    model, losses = sft_train(
        model, INSTRUCTION_DATA, num_epochs=250, lr=lr, use_lora=use_lora
    )

    print()
    print("POST-SFT: fine-tuned model loss on raw text")
    sft_loss = measure_forgetting(model, test_text)
    change = (sft_loss - base_loss) / base_loss * 100
    print(f"  SFT model loss: {sft_loss:.4f}")
    print(f"  Change vs base: {change:+.1f}%  (some forgetting is EXPECTED now that")
    print("  the model actually learned the instruction distribution)\n")

    print("=" * 70)
    print("INSTRUCTION FOLLOWING EVALUATION")
    print("=" * 70)
    print()
    evaluate_instruction_following(model, [
        "What is the capital of France?",
        "Name three programming languages.",
        "Explain gravity in one sentence.",
    ])

    print("=" * 70)
    print("TRAINING LOSS CURVE (start -> end)")
    print("=" * 70)
    print()
    print(f"  First epoch loss: {losses[0]:.4f}")
    print(f"  Last  epoch loss: {losses[-1]:.4f}")
    print(f"  Reduction:        {(1 - losses[-1] / losses[0]) * 100:.1f}%")
