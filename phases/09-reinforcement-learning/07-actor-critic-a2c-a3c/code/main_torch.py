"""A2C-style actor-critic with GAE on 4x4 GridWorld — NumPy/PyTorch port of main.py.

Same environment, linear actor (softmax over 16 one-hot features -> 4 actions,
no bias), linear critic (value head, no bias, zero-init), GAE(gamma, lam)
advantage estimation, advantage normalization, entropy bonus and hyperparameters
as the pure-Python main.py. The hand-written softmax/sampling, the manual
policy-gradient + value-regression + entropy updates are replaced by two
nn.Linear heads, torch.distributions.Categorical, and autograd on:

    actor_loss  = -sum_t  adv_norm_t * log pi(a_t|s_t)   - ent_coef * sum_t H(pi(.|s_t))
    critic_loss =  0.5 * sum_t (return_t - V(s_t))^2

Two notes on faithfulness:
  * The critic target (`return_t`) and the GAE advantages use critic values
    snapshotted during the rollout (detached), exactly as main.py does — the
    value used in GAE is frozen, only the regression target flows gradient.
  * main.py hand-codes an approximate entropy gradient; here we use the exact
    categorical entropy via autograd. Learning curve and final policy match;
    per-number values differ (as with the other torch ports: different init
    and RNG stream).
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
    """Linear actor (theta) + linear critic (w), both bias-free, sharing the
    one-hot input — the two weight matrices main.py updates by hand."""

    def __init__(self, n_feat, n_actions):
        super().__init__()
        self.actor = nn.Linear(n_feat, n_actions, bias=False)
        self.critic = nn.Linear(n_feat, 1, bias=False)
        nn.init.normal_(self.actor.weight, std=0.1)   # matches init_theta
        nn.init.zeros_(self.critic.weight)            # matches init_w

    def forward(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)


def rollout(model, rng, max_steps=100):
    """Sample a trajectory; snapshot the critic value at each visited state
    (detached, like main.py's `value(w, x)` computed during the rollout)."""
    traj = []
    s = reset()
    for _ in range(max_steps):
        x = features(s)
        with torch.no_grad():
            logits, v = model(torch.from_numpy(x).unsqueeze(0))
            probs = torch.softmax(logits.squeeze(0), dim=0)
        a = int(torch.multinomial(probs, 1, generator=rng).item())
        s_next, r, done = step(s, a)
        traj.append({"x": x, "a": a, "r": r, "v": float(v.item()), "done": done})
        if done:
            break
        s = s_next
    return traj


def gae_advantages(traj, gamma=0.99, lam=0.95):
    T = len(traj)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_v = 0.0 if traj[t]["done"] else (traj[t + 1]["v"] if t + 1 < T else 0.0)
        delta = traj[t]["r"] + gamma * next_v - traj[t]["v"]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + traj[t]["v"] for t, a in enumerate(advantages)]
    return advantages, returns


def normalize(xs):
    t = torch.tensor(xs, dtype=torch.float32)
    if t.numel() < 2:
        return t
    return (t - t.mean()) / (t.std(unbiased=False) + 1e-8)


def actor_critic(episodes, lr_a=0.05, lr_v=0.1, gamma=0.99, lam=0.95, ent_coef=0.01, seed=0):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = ActorCritic(N_FEAT, N_ACTIONS)
    # Separate optimizers keep the actor/critic learning rates distinct (0.05 vs 0.1).
    opt_actor = torch.optim.SGD(model.actor.parameters(), lr=lr_a)
    opt_critic = torch.optim.SGD(model.critic.parameters(), lr=lr_v)
    returns_log = []

    for _ in range(episodes):
        traj = rollout(model, rng)
        advs, returns = gae_advantages(traj, gamma=gamma, lam=lam)
        adv_norm = normalize(advs)

        X = torch.from_numpy(np.stack([n["x"] for n in traj]))
        A = torch.tensor([n["a"] for n in traj], dtype=torch.long)
        R = torch.tensor(returns, dtype=torch.float32)

        logits, values = model(X)
        logp = torch.log_softmax(logits, dim=1)
        chosen_logp = logp.gather(1, A.unsqueeze(1)).squeeze(1)
        entropy = -(logp.exp() * logp).sum(dim=1)

        actor_loss = -(adv_norm * chosen_logp).sum() - ent_coef * entropy.sum()
        critic_loss = 0.5 * ((R - values) ** 2).sum()

        opt_actor.zero_grad()
        opt_critic.zero_grad()
        (actor_loss + critic_loss).backward()
        opt_actor.step()
        opt_critic.step()

        if traj:
            mc_return = 0.0
            for r in reversed([n["r"] for n in traj]):
                mc_return = r + gamma * mc_return
            returns_log.append(mc_return)

    return model, returns_log


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


def block_mean(xs, block):
    return [sum(xs[i : i + block]) / block for i in range(0, len(xs) - block + 1, block)]


def critic_value(model, state):
    with torch.no_grad():
        _, v = model(torch.from_numpy(features(state)).unsqueeze(0))
    return float(v.item())


def main():
    episodes = 1500
    model, log = actor_critic(episodes, lam=0.95, seed=7)

    print("=== A2C-style actor-critic (torch) with GAE(lam=0.95) on 4x4 GridWorld ===")
    print()
    print("learning curve (mean return per 150 episodes):")
    for i, m in enumerate(block_mean(log, 150)):
        print(f"  block {i+1}: mean return = {m:6.2f}")

    print()
    print_policy(greedy_policy(model), "greedy policy from actor")
    print()
    print("critic values V_phi(s):")
    for r in range(GRID):
        row = " ".join(f"{critic_value(model, (r, c)):7.2f}" for c in range(GRID))
        print("   " + row)

    print()
    print(f"final mean return (last 150 eps) = {sum(log[-150:]) / 150:.2f}  (optimal = -6.0)")


if __name__ == "__main__":
    main()
