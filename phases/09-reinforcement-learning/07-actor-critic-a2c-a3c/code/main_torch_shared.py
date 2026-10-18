"""Shared-backbone actor-critic on 4x4 GridWorld — the modern A2C architecture.

Companion to main_torch.py (which uses TWO fully separate linear networks).
Here the actor and critic share a trunk `phi` and branch into two small heads:

                   +-> actor_head  -> logits
    state -> phi --+
                   +-> critic_head -> value

Consequences, and the whole point of this example:
  * The backbone parameters are reached by BOTH losses. In a single backward()
    autograd sums the actor gradient and the (c_v-scaled) critic gradient at the
    shared activations, then propagates that COMBINED signal through the trunk.
  * You can no longer give the actor and critic independent learning rates on
    the shared part, so there is ONE optimizer and a value-loss coefficient c_v
    becomes the knob that balances the two signals.
  * Gradient clipping is added (standard in PPO/A2C) because the two summed
    gradients can spike early in training.

Everything else — env, GAE, advantage normalization, entropy bonus, detach
boundaries — is identical to main_torch.py so the two files diff cleanly.
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


class SharedActorCritic(nn.Module):
    """Shared trunk + two heads. The trunk's parameters are the ones that
    receive gradient from BOTH the actor and the critic loss."""

    def __init__(self, n_feat, n_actions, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(          # shared phi
            nn.Linear(n_feat, hidden),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, n_actions)   # policy head only
        self.critic_head = nn.Linear(hidden, 1)          # value head only

    def forward(self, x):
        h = self.backbone(x)                    # both heads read the same h
        return self.actor_head(h), self.critic_head(h).squeeze(-1)


def rollout(model, rng, max_steps=100):
    """Sample a trajectory; snapshot the critic value at each visited state
    (detached, so GAE advantages and returns are constants w.r.t. autograd)."""
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


def actor_critic(episodes, lr=0.003, gamma=0.99, lam=0.95,
                 value_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, seed=0):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = SharedActorCritic(N_FEAT, N_ACTIONS)
    # ONE optimizer over ALL params (trunk + both heads). With a shared trunk
    # you cannot hand the actor and critic separate learning rates on it, so the
    # balance between the two objectives is controlled by value_coef instead.
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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

        actor_loss = -(adv_norm * chosen_logp).mean()
        critic_loss = 0.5 * ((R - values) ** 2).mean()
        entropy_bonus = entropy.mean()

        # Single scalar. backward() sends actor-grad and value_coef*critic-grad
        # back to the trunk, where autograd SUMS them before descending phi.
        loss = actor_loss + value_coef * critic_loss - ent_coef * entropy_bonus

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()

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


def report_gradient_sharing(model, seed=7):
    """Make the shared-gradient claim concrete: run one episode, backprop the
    actor loss and the critic loss SEPARATELY, and show that both deposit a
    non-zero gradient into the SAME backbone weight — their sum is what the
    real update uses."""
    rng = torch.Generator().manual_seed(seed)
    traj = rollout(model, rng)
    advs, returns = gae_advantages(traj)
    adv_norm = normalize(advs)
    X = torch.from_numpy(np.stack([n["x"] for n in traj]))
    A = torch.tensor([n["a"] for n in traj], dtype=torch.long)
    R = torch.tensor(returns, dtype=torch.float32)

    logits, values = model(X)
    logp = torch.log_softmax(logits, dim=1)
    chosen = logp.gather(1, A.unsqueeze(1)).squeeze(1)
    actor_loss = -(adv_norm * chosen).mean()
    critic_loss = 0.5 * ((R - values) ** 2).mean()

    trunk_w = model.backbone[0].weight

    model.zero_grad()
    actor_loss.backward(retain_graph=True)
    g_actor = trunk_w.grad.norm().item()

    model.zero_grad()
    critic_loss.backward()
    g_critic = trunk_w.grad.norm().item()

    print("gradient reaching the SHARED trunk weight (L2 norm):")
    print(f"  from actor_loss  only = {g_actor:.4f}")
    print(f"  from critic_loss only = {g_critic:.4f}")
    print("  -> both are non-zero, so the real update descends their SUM.")
    print("     (actor_head / critic_head, by contrast, get gradient from one loss each.)")
    print(f"  actor_head grad from critic_loss = "
          f"{(model.actor_head.weight.grad.norm().item() if model.actor_head.weight.grad is not None else 0.0):.4f}")


def main():
    episodes = 1500
    model, log = actor_critic(episodes, lam=0.95, seed=7)

    print("=== Shared-backbone actor-critic (torch) with GAE(lam=0.95) on 4x4 GridWorld ===")
    print()
    print("learning curve (mean return per 150 episodes):")
    for i, m in enumerate(block_mean(log, 150)):
        print(f"  block {i+1}: mean return = {m:6.2f}")

    print()
    print_policy(greedy_policy(model), "greedy policy from actor head")
    print()
    print("critic values V_phi(s):")
    for r in range(GRID):
        row = " ".join(f"{critic_value(model, (r, c)):7.2f}" for c in range(GRID))
        print("   " + row)

    print()
    print(f"final mean return (last 150 eps) = {sum(log[-150:]) / 150:.2f}  (optimal = -6.0)")
    print()
    report_gradient_sharing(model)


if __name__ == "__main__":
    main()
