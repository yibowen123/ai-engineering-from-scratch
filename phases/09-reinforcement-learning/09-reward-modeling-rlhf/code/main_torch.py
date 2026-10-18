"""Reward modeling + PPO-RLHF — NumPy/PyTorch port of main.py.

Two stages, same toy setup as main.py:

  Stage 1 — Reward model. A linear scorer over a 12-word bag-of-words vocab,
  trained on preference pairs with the Bradley-Terry / pairwise-logistic loss
      L = -log sigmoid( score(y_pos) - score(y_neg) ).
  main.py hand-codes the gradient (scale `1 - p`, push pos tokens up / neg
  down); here it's an nn.Linear + autograd on softplus(-(diff)).

  Stage 2 — PPO-RLHF. A tiny one-token "policy" theta[prompt, vocab] is
  optimized against the frozen reward model with a KL-to-reference penalty
  folded into the reward:  reward = RM(token) - beta * KL(pi_new || pi_ref).
  Rewards are whitened into advantages and the policy takes a PPO clipped step.
  main.py hand-codes the clipped policy-gradient (skipping clipped samples);
  here it's the standard surrogate min(ratio*A, clip(ratio,1±eps)*A) via autograd
  (a clipped sample contributes zero gradient, same as main.py's skip).

As with the other torch ports, per-number values differ (init + RNG stream),
but the RM weights, accuracy, and the beta-sweep trend (bigger beta -> lower
final KL) match.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROMPTS = ("help me", "answer me", "explain this")
GOOD = ("clear", "specific", "kind", "thorough", "precise", "helpful")
BAD = ("vague", "rude", "wrong", "short", "cold", "careless")
VOCAB = tuple(sorted(set(GOOD + BAD)))
TOK2IDX = {t: i for i, t in enumerate(VOCAB)}
N_VOCAB = len(VOCAB)
N_PROMPTS = len(PROMPTS)


def bag_vector(tokens):
    """Bag-of-words count vector over VOCAB — the feature main.py scores with."""
    v = np.zeros(N_VOCAB, dtype=np.float32)
    for t in tokens:
        v[TOK2IDX[t]] += 1.0
    return v


def sample_pair(rng):
    p = int(torch.randint(len(PROMPTS), (1,), generator=rng).item())
    gi = torch.randint(len(GOOD), (2,), generator=rng).tolist()
    bi = torch.randint(len(BAD), (2,), generator=rng).tolist()
    y_pos = (GOOD[gi[0]], GOOD[gi[1]])
    y_neg = (BAD[bi[0]], BAD[bi[1]])
    return PROMPTS[p], y_pos, y_neg


# ----------------------------- Stage 1: reward model -----------------------------

class RewardModel(nn.Module):
    """Linear scorer over the bag-of-words vocab — main.py's weight dict `w`."""

    def __init__(self, n_vocab):
        super().__init__()
        self.w = nn.Linear(n_vocab, 1, bias=False)
        nn.init.zeros_(self.w.weight)   # main.py's defaultdict(float) starts at 0

    def forward(self, bow):
        return self.w(bow).squeeze(-1)


def train_rm(n_pairs=600, lr=0.1, seed=0):
    rng = torch.Generator().manual_seed(seed)
    rm = RewardModel(N_VOCAB)
    opt = torch.optim.SGD(rm.parameters(), lr=lr)
    for _ in range(n_pairs):
        _, y_pos, y_neg = sample_pair(rng)
        pos = torch.from_numpy(bag_vector(y_pos)).unsqueeze(0)
        neg = torch.from_numpy(bag_vector(y_neg)).unsqueeze(0)
        diff = rm(pos) - rm(neg)
        # -log sigmoid(diff) == softplus(-diff); gradient scale is (1 - p), exactly main.py.
        loss = F.softplus(-diff).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return rm


def rm_accuracy(rm, n_pairs=200, seed=1):
    rng = torch.Generator().manual_seed(seed)
    correct = 0
    with torch.no_grad():
        for _ in range(n_pairs):
            _, y_pos, y_neg = sample_pair(rng)
            pos = torch.from_numpy(bag_vector(y_pos)).unsqueeze(0)
            neg = torch.from_numpy(bag_vector(y_neg)).unsqueeze(0)
            if rm(pos).item() > rm(neg).item():
                correct += 1
    return correct / n_pairs


# ----------------------------- Stage 2: PPO-RLHF -----------------------------

def kl_to_uniform(logits):
    """KL(pi_new || pi_ref) with the reference being the zero-logit (uniform)
    policy — matches main.py's reference initialized to all-zeros."""
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    log_ref = -np.log(N_VOCAB)  # log(1/V)
    return (p * (logp - log_ref)).sum(dim=-1)


def rlhf_loop(rm, updates=150, beta=0.1, lr=0.05, eps=0.2, seed=0):
    rng = torch.Generator().manual_seed(seed)
    theta = nn.Parameter(torch.zeros(N_PROMPTS, N_VOCAB))
    opt = torch.optim.SGD([theta], lr=lr)
    rm_w = rm.w.weight.detach().squeeze(0)   # per-token reward, frozen

    history = []
    for it in range(updates):
        # --- collect 16 one-token rollouts under the current policy (no grad) ---
        p_idx, tokens, logp_old, rewards, kls = [], [], [], [], []
        with torch.no_grad():
            for _ in range(16):
                pi = int(torch.randint(N_PROMPTS, (1,), generator=rng).item())
                logits = theta[pi]
                probs = F.softmax(logits, dim=-1)
                tok = int(torch.multinomial(probs, 1, generator=rng).item())
                kl_term = kl_to_uniform(logits)
                reward = rm_w[tok] - beta * kl_term
                p_idx.append(pi)
                tokens.append(tok)
                logp_old.append(float(torch.log(probs[tok].clamp_min(1e-12)).item()))
                rewards.append(float(reward.item()))
                kls.append(float(kl_term.item()))

        # --- whiten rewards into advantages (bandit: no value net / GAE) ---
        r = torch.tensor(rewards)
        advs = (r - r.mean()) / (r.std(unbiased=False) + 1e-8)

        # --- one PPO clipped step over the batch ---
        P = torch.tensor(p_idx, dtype=torch.long)
        A = torch.tensor(tokens, dtype=torch.long)
        lpo = torch.tensor(logp_old)
        logp = F.log_softmax(theta[P], dim=-1).gather(1, A.unsqueeze(1)).squeeze(1)
        ratio = torch.exp(logp - lpo)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advs
        loss = -torch.min(surr1, surr2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        mean_kl = sum(kls) / len(kls)
        mean_rm = sum(rw + beta * k for rw, k in zip(rewards, kls)) / len(rewards)
        history.append((it + 1, mean_rm, mean_kl))
    return theta.detach(), history


def main():
    rm = train_rm(n_pairs=600, seed=42)

    print("=== Stage 1: reward model (Bradley-Terry pairwise logistic) ===")
    print()
    w = {t: rm.w.weight.detach()[0, i].item() for t, i in TOK2IDX.items()}
    print("top positive-weight tokens:")
    for tok in sorted(w, key=lambda t: -w[t])[:6]:
        print(f"  {tok:<10} w = {w[tok]:+.3f}")
    print()
    print("top negative-weight tokens:")
    for tok in sorted(w, key=lambda t: w[t])[:6]:
        print(f"  {tok:<10} w = {w[tok]:+.3f}")
    print()
    print(f"RM pairwise accuracy on holdout (200 pairs) = {rm_accuracy(rm):.3f}")
    print()

    print("=== Stage 2: PPO-RLHF against RM with KL penalty ===")
    print()
    for beta in (0.01, 0.1, 1.0):
        _, hist = rlhf_loop(rm, updates=150, beta=beta, seed=0)
        first, last = hist[0], hist[-1]
        print(f"beta={beta:<5}  initial: RM={first[1]:+.3f} KL={first[2]:.3f}   "
              f"final: RM={last[1]:+.3f} KL={last[2]:.3f}")


if __name__ == "__main__":
    main()
