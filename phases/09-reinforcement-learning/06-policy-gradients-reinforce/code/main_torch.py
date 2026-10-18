"""REINFORCE on 4x4 GridWorld — NumPy/PyTorch port of main.py.

Same environment, policy parameterization (linear softmax over 16 one-hot
features -> 4 actions, no bias), returns-to-go, optional EMA baseline and
hyperparameters as the pure-Python main.py. The only difference is mechanical:
the hand-written softmax, sampling and per-feature policy-gradient update are
replaced by an nn.Module, torch.distributions.Categorical, and autograd on the
REINFORCE loss  L = -sum_t advantage_t * log pi(a_t | s_t).

Numbers won't match main.py bit-for-bit — torch's weight init and RNG stream
differ from random.Random — but the learning curves and final policy do.
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


class PolicyNet(nn.Module):
    """Linear softmax policy: logits = theta @ x, no bias — the `logits` +
    `softmax` that main.py computes by hand, as a single Linear layer."""

    def __init__(self, n_feat, n_actions):
        super().__init__()
        self.theta = nn.Linear(n_feat, n_actions, bias=False)

    def forward(self, x):
        return self.theta(x)


def rollout(policy, rng, max_steps=100):
    """Sample a trajectory under the current policy. Mirrors main.py's rollout:
    softmax over logits, categorical sample, step until terminal."""
    traj = []
    s = reset()
    for _ in range(max_steps):
        x = features(s)
        with torch.no_grad():
            logits = policy(torch.from_numpy(x).unsqueeze(0)).squeeze(0)
            probs = torch.softmax(logits, dim=0)
        a = int(torch.multinomial(probs, 1, generator=rng).item())
        s_next, r, done = step(s, a)
        traj.append((x, a, r))
        if done:
            break
        s = s_next
    return traj


def returns_to_go(traj, gamma):
    G = 0.0
    out = []
    for _, _, r in reversed(traj):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return out


def reinforce(episodes, lr=0.05, gamma=0.99, use_baseline=False, seed=0):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    policy = PolicyNet(N_FEAT, N_ACTIONS)
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    baseline = 0.0
    returns_log = []
    for _ in range(episodes):
        traj = rollout(policy, rng)
        returns = returns_to_go(traj, gamma)
        if use_baseline:
            baseline = 0.95 * baseline + 0.05 * returns[0]

        # Vectorized REINFORCE loss over the whole trajectory:
        #   L = -sum_t advantage_t * log pi(a_t | s_t)
        # autograd reproduces main.py's per-feature update  (1[i==a] - p_i) * x_j.
        X = torch.from_numpy(np.stack([x for x, _, _ in traj]))
        A = torch.tensor([a for _, a, _ in traj], dtype=torch.long)
        G = torch.tensor(returns, dtype=torch.float32)
        adv = G - (baseline if use_baseline else 0.0)

        logp = torch.log_softmax(policy(X), dim=1)
        chosen_logp = logp.gather(1, A.unsqueeze(1)).squeeze(1)
        loss = -(adv * chosen_logp).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        returns_log.append(returns[0] if returns else 0.0)
    return policy, returns_log


def greedy_policy(policy):
    out = {}
    for r in range(GRID):
        for c in range(GRID):
            if (r, c) == TERMINAL:
                continue
            with torch.no_grad():
                z = policy(torch.from_numpy(features((r, c))).unsqueeze(0)).squeeze(0)
            out[(r, c)] = ACTIONS[int(torch.argmax(z).item())]
    return out


def print_policy(policy_map, title):
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    print(f"  {title}")
    for r in range(GRID):
        row = []
        for c in range(GRID):
            if (r, c) == TERMINAL:
                row.append(".")
            elif (r, c) in policy_map:
                row.append(arrows[policy_map[(r, c)]])
            else:
                row.append("?")
        print("   " + " ".join(row))


def block_mean(xs, block):
    return [sum(xs[i : i + block]) / block for i in range(0, len(xs) - block + 1, block)]


def main():
    episodes = 2000
    _, r_plain = reinforce(episodes, use_baseline=False, seed=42)
    policy_b, r_base = reinforce(episodes, use_baseline=True, seed=42)

    print(f"=== REINFORCE (torch) on 4x4 GridWorld, {episodes} episodes, lr=0.05, gamma=0.99 ===")
    print()
    print("learning curves (mean return per 200 episodes):")
    for i, (a, b) in enumerate(zip(block_mean(r_plain, 200), block_mean(r_base, 200))):
        print(f"  block {i+1}: vanilla={a:7.2f}   with-baseline={b:7.2f}")

    print()
    print_policy(greedy_policy(policy_b), "final greedy policy (with-baseline)")
    print()
    print(f"final mean return (last 200 eps): vanilla={sum(r_plain[-200:])/200:.2f}   with-baseline={sum(r_base[-200:])/200:.2f}")
    print("(optimal return on this 4x4 GridWorld = -6.0)")


if __name__ == "__main__":
    main()
