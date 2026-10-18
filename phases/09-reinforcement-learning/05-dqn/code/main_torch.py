"""DQN on 4x4 GridWorld — NumPy/PyTorch port of main.py.

Same environment, network shape (16->32->4), replay buffer, target network,
epsilon schedule and hyperparameters as the pure-Python main.py. The only
difference is mechanical: the hand-written forward/backward pass in
`train_step` is replaced by an nn.Module + autograd + an SGD optimizer, and
the minibatch is vectorized into tensors instead of looped sample-by-sample.
Run it next to main.py and compare the learning curves.
"""

import random

import numpy as np
import torch
import torch.nn as nn


GRID = 4
TERMINAL = (3, 3)
ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def reset():
    return (0, 0)


def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = DELTAS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL


def state_features(state):
    """One-hot encoding of the grid cell — same 16-dim features as main.py."""
    feat = np.zeros(GRID * GRID, dtype=np.float32)
    r, c = state
    feat[r * GRID + c] = 1.0
    return feat


class QNet(nn.Module):
    """One hidden layer, ReLU, linear head — the MLP main.py builds by hand."""

    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def epsilon_greedy(net, state, rng, epsilon):
    if rng.random() < epsilon:
        return rng.randrange(len(ACTIONS))
    with torch.no_grad():
        x = torch.from_numpy(state_features(state)).unsqueeze(0)
        q = net(x).squeeze(0)
    return int(torch.argmax(q).item())


def train_step(online, target, optimizer, batch, gamma):
    """One gradient step on a minibatch.

    Vectorized equivalent of main.py's per-sample loop: build tensors for the
    batch, compute Q(s,a) from the online net and the TD target r + gamma *
    max_a' Q_target(s', a') (zeroed on terminal), then take a Huber/MSE step.
    main.py used 0.5 * td_error^2, i.e. plain MSE, so we use MSELoss here.
    """
    states = torch.from_numpy(np.stack([state_features(s) for s, _, _, _, _ in batch]))
    actions = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.long)
    rewards = torch.tensor([r for _, _, r, _, _ in batch], dtype=torch.float32)
    next_states = torch.from_numpy(np.stack([state_features(sn) for _, _, _, sn, _ in batch]))
    dones = torch.tensor([done for _, _, _, _, done in batch], dtype=torch.float32)

    # Q(s,a) for the taken actions.
    q = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # TD target from the frozen target net; no grad flows through it.
    with torch.no_grad():
        max_next = target(next_states).max(dim=1).values
        y = rewards + gamma * max_next * (1.0 - dones)

    loss = nn.functional.mse_loss(q, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    rng = random.Random(0)
    torch.manual_seed(0)

    n_in = GRID * GRID
    online = QNet(n_in, 32, len(ACTIONS))
    target = QNet(n_in, 32, len(ACTIONS))
    target.load_state_dict(online.state_dict())

    buffer = []
    capacity = 2000
    batch = 32
    gamma = 0.99
    lr = 0.05
    sync_every = 200
    episodes = 400
    step_count = 0

    optimizer = torch.optim.SGD(online.parameters(), lr=lr)

    returns_log = []
    for ep in range(episodes):
        s = reset()
        total = 0.0
        epsilon = max(0.05, 1.0 - ep / 200)
        for _ in range(50):
            a = epsilon_greedy(online, s, rng, epsilon)
            s_next, r, done = step(s, ACTIONS[a])
            total += r
            buffer.append((s, a, r, s_next, done))
            if len(buffer) > capacity:
                buffer.pop(0)
            if len(buffer) >= batch:
                mb = rng.sample(buffer, batch)
                train_step(online, target, optimizer, mb, gamma)
            step_count += 1
            if step_count % sync_every == 0:
                target.load_state_dict(online.state_dict())
            if done:
                break
            s = s_next
        returns_log.append(total)

    print(f"=== DQN (torch) on 4x4 GridWorld ({episodes} episodes, batch={batch}, target sync every {sync_every} steps) ===")
    print()
    print("learning curve (mean return per block of 50 episodes):")
    for i in range(0, episodes, 50):
        chunk = returns_log[i : i + 50]
        print(f"  episodes {i:3d}-{i+49:3d}: mean = {sum(chunk) / len(chunk):6.2f}")

    print()
    with torch.no_grad():
        q0 = online(torch.from_numpy(state_features((3, 2))).unsqueeze(0)).squeeze(0)
    print("Q(0,0) per action:")
    for a, v in zip(ACTIONS, q0.tolist()):
        print(f"  {a:<6} = {v:6.2f}")
    print()
    print("greedy policy from trained net:")
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    for r in range(GRID):
        row = []
        for c in range(GRID):
            if (r, c) == TERMINAL:
                row.append(".")
                continue
            with torch.no_grad():
                q = online(torch.from_numpy(state_features((r, c))).unsqueeze(0)).squeeze(0)
            best = ACTIONS[int(torch.argmax(q).item())]
            row.append(arrows[best])
        print("   " + " ".join(row))


if __name__ == "__main__":
    main()
