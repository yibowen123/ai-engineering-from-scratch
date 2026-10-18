"""PPO on 4x4 GridWorld — NumPy/PyTorch port of main.py.

Same environment, linear actor (softmax over 16 one-hot features -> 4 actions,
no bias) and linear critic (value head, no bias, zero-init), vectorized rollout
collection across 8 "envs", GAE(gamma, lam), per-minibatch advantage
normalization, K epochs of minibatch updates with the PPO clipped surrogate,
and the same KL / clip-fraction diagnostics as the pure-Python main.py.

What is mechanical vs hand-written in main.py:
  * main.py hand-codes the clipped policy-gradient (it zeroes `pg_scale` when a
    sample is in the clipping region) and the value regression. Here we write
    the standard clipped surrogate
        L^CLIP = min( ratio * A, clip(ratio, 1-eps, 1+eps) * A )
    and let autograd produce exactly that sub-gradient (ratio*A when unclipped,
    0 in the clipped region).
  * main.py applies per-sample SGD inside each minibatch; this port takes one
    gradient step per minibatch on the summed loss (closest magnitude match),
    with two optimizers so the actor (lr_a) and critic (lr_v) keep distinct
    learning rates.

No entropy bonus — main.py has none, so this stays faithful. As with the other
torch ports, per-number values differ (init + RNG stream), but the learning
curve, diagnostics and final policy match.
"""

import numpy as np
import torch
import torch.nn as nn


GRID = 4
TERMINAL = (3, 3)
ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
N_ACTIONS = len(ACTIONS)
N_FEAT = GRID * GRID


def reset():
    return (0, 0)


def step(state, action_idx):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = DELTAS[ACTIONS[action_idx]]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL


def features(state):
    """One-hot encoding of the grid cell — same 16-dim features as main.py."""
    x = np.zeros(N_FEAT, dtype=np.float32)
    r, c = state
    x[r * GRID + c] = 1.0
    return x


class ActorCritic(nn.Module):
    """Linear actor (theta) + linear critic (w), both bias-free — the two weight
    matrices main.py updates by hand."""

    def __init__(self, n_feat, n_actions):
        super().__init__()
        self.actor = nn.Linear(n_feat, n_actions, bias=False)
        self.critic = nn.Linear(n_feat, 1, bias=False)
        nn.init.normal_(self.actor.weight, std=0.1)   # matches init_theta
        nn.init.zeros_(self.critic.weight)            # matches init_w

    def forward(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)


def collect_rollout(model, rng, horizon=50, n_envs=8):
    """Run the CURRENT policy to gather transitions. v_old and log_pi_old are
    snapshotted here (no grad) — they are the fixed references PPO's ratio and
    value target are measured against."""
    buffer = []
    for _ in range(n_envs):
        s = reset()
        for _ in range(horizon):
            x = features(s)
            with torch.no_grad():
                logits, v = model(torch.from_numpy(x).unsqueeze(0))
                logp = torch.log_softmax(logits.squeeze(0), dim=0)
                probs = logp.exp()
            a = int(torch.multinomial(probs, 1, generator=rng).item())
            s_next, r, done = step(s, a)
            buffer.append({
                "x": x,
                "a": a,
                "r": r,
                "done": done,
                "v_old": float(v.item()),
                "log_pi_old": float(logp[a].item()),
            })
            if done:
                break
            s = s_next
    return buffer


def gae(buffer, gamma=0.99, lam=0.95):
    T = len(buffer)
    advantages = [0.0] * T
    gae_val = 0.0
    for t in reversed(range(T)):
        next_v = 0.0 if buffer[t]["done"] else (buffer[t + 1]["v_old"] if t + 1 < T else 0.0)
        delta = buffer[t]["r"] + gamma * next_v - buffer[t]["v_old"]
        gae_val = delta + gamma * lam * gae_val
        advantages[t] = gae_val
    returns = [a + buffer[t]["v_old"] for t, a in enumerate(advantages)]
    return advantages, returns


def normalize(t):
    if t.numel() < 2:
        return t
    return (t - t.mean()) / (t.std(unbiased=False) + 1e-8)


def ppo_update(model, opt_actor, opt_critic, buffer, advantages, returns,
               eps=0.2, epochs=4, batch=32, rng=None):
    rng = rng or torch.Generator().manual_seed(0)
    N = len(buffer)
    X_all = torch.from_numpy(np.stack([rec["x"] for rec in buffer]))
    A_all = torch.tensor([rec["a"] for rec in buffer], dtype=torch.long)
    logp_old_all = torch.tensor([rec["log_pi_old"] for rec in buffer], dtype=torch.float32)
    adv_all = torch.tensor(advantages, dtype=torch.float32)
    ret_all = torch.tensor(returns, dtype=torch.float32)

    kl_total, kl_count, clip_hits, total = 0.0, 0, 0, 0

    for _ in range(epochs):
        perm = torch.randperm(N, generator=rng)
        for i in range(0, N, batch):
            idx = perm[i : i + batch]
            X, A = X_all[idx], A_all[idx]
            logp_old = logp_old_all[idx]
            adv = normalize(adv_all[idx])          # per-minibatch, like main.py
            ret = ret_all[idx]

            logits, values = model(X)
            logp = torch.log_softmax(logits, dim=1).gather(1, A.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(logp - logp_old)

            # PPO clipped surrogate — autograd gives ratio*A unclipped, 0 clipped.
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
            policy_loss = -torch.min(surr1, surr2).sum()
            value_loss = 0.5 * ((ret - values) ** 2).sum()

            opt_actor.zero_grad()
            opt_critic.zero_grad()
            (policy_loss + value_loss).backward()
            opt_actor.step()
            opt_critic.step()

            # Diagnostics (no grad): match main.py's definitions exactly.
            with torch.no_grad():
                clipped = ((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps))
                clip_hits += int(clipped.sum().item())
                total += adv.numel()
                kl_total += float((logp_old - logp).sum().item())
                kl_count += adv.numel()

    return kl_total / max(1, kl_count), clip_hits / max(1, total)


def greedy_policy(model):
    policy = {}
    for r in range(GRID):
        for c in range(GRID):
            if (r, c) == TERMINAL:
                continue
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(features((r, c))).unsqueeze(0))
            policy[(r, c)] = ACTIONS[int(torch.argmax(logits.squeeze(0)).item())]
    return policy


def print_policy(policy, title):
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    print(f"  {title}")
    for r in range(GRID):
        row = []
        for c in range(GRID):
            if (r, c) == TERMINAL:
                row.append(".")
            elif (r, c) in policy:
                row.append(arrows[policy[(r, c)]])
            else:
                row.append("?")
        print("   " + " ".join(row))


def evaluate(model, rng, episodes=50):
    total = 0.0
    for _ in range(episodes):
        s = reset()
        ep_total = 0.0
        for _ in range(50):
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(features(s)).unsqueeze(0))
                probs = torch.softmax(logits.squeeze(0), dim=0)
            a = int(torch.multinomial(probs, 1, generator=rng).item())
            s, r, done = step(s, a)
            ep_total += r
            if done:
                break
        total += ep_total
    return total / episodes


def main():
    torch.manual_seed(123)
    rng = torch.Generator().manual_seed(123)
    model = ActorCritic(N_FEAT, N_ACTIONS)
    opt_actor = torch.optim.SGD(model.actor.parameters(), lr=0.05)
    opt_critic = torch.optim.SGD(model.critic.parameters(), lr=0.1)

    updates = 60
    print(f"=== PPO (torch) on 4x4 GridWorld ({updates} updates, 8 envs x 50 steps, eps=0.2, K=4) ===")
    print()

    for it in range(updates):
        buffer = collect_rollout(model, rng)
        advantages, returns = gae(buffer)
        mean_kl, clip_frac = ppo_update(model, opt_actor, opt_critic, buffer, advantages, returns, rng=rng)
        if (it + 1) % 10 == 0:
            mean_ret = evaluate(model, torch.Generator().manual_seed(it))
            print(f"  update {it+1:3d}  mean_return={mean_ret:6.2f}  mean_KL={mean_kl:+.4f}  clip_frac={clip_frac:.3f}")

    print()
    print_policy(greedy_policy(model), "greedy policy")
    final = evaluate(model, torch.Generator().manual_seed(999), episodes=200)
    print()
    print(f"final evaluated return (200 episodes) = {final:.2f}  (optimal = -6.0)")


if __name__ == "__main__":
    main()
