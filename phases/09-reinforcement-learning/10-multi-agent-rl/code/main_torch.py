"""Multi-agent RL on a cooperative 2-agent GridWorld — PyTorch / deep-RL version.

main.py solves this with TABULAR Q-learning (dict Q-tables, no gradients). A
mechanical "tensorize the dict" port would add nothing, so this file instead
renders the SAME comparison with function approximation — the direction the
lesson's closing note points at (CTDE, MAPPO, QMIX are all neural). It reuses
the DQN machinery from lesson 05 (replay buffer + target network + epsilon
decay) and contrasts the two tabular ideas in their deep-RL form:

  * Independent DQN (IDQN)  <- main.py's `independent_q`
        one Q-network per agent, each outputs 4 action-values; each agent acts
        and bootstraps from its OWN network. Decentralized; non-stationary from
        each agent's view (the other agent keeps changing).

  * Joint-action DQN        <- main.py's `joint_q_learning`
        a single Q-network outputting |A|^2 = 16 joint-action values. Correct
        global view, but the output layer grows exponentially in the number of
        agents — exactly the scaling problem the note warns about.

Same env (5x5, shared reward +10 on joint goal else -1), same greedy evaluation
and the same concluding note. State is the two agents' positions as two
concatenated one-hot vectors (25 + 25 = 50 dims), the lossless analogue of
main.py's dict key.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


GRID = 5
GOAL = (4, 4)
ACTIONS = ("up", "down", "left", "right")
DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
N_ACTIONS = len(ACTIONS)
JOINT_ACTIONS = [(a, b) for a in range(N_ACTIONS) for b in range(N_ACTIONS)]
N_JOINT = len(JOINT_ACTIONS)
N_FEAT = 2 * GRID * GRID   # one-hot(pos1) ++ one-hot(pos2)


def move(pos, action_idx):
    dr, dc = DELTAS[ACTIONS[action_idx]]
    r, c = pos
    return (min(max(r + dr, 0), GRID - 1), min(max(c + dc, 0), GRID - 1))


def reset():
    return ((0, 0), (GRID - 1, 0))


def step(state, action_pair):
    a1_pos, a2_pos = state
    new1 = move(a1_pos, action_pair[0])
    new2 = move(a2_pos, action_pair[1])
    done = (new1 == GOAL) and (new2 == GOAL)
    reward = 10.0 if done else -1.0
    return (new1, new2), reward, done


def features(state):
    """Two agent positions as concatenated one-hot vectors (lossless key)."""
    (r1, c1), (r2, c2) = state
    x = np.zeros(N_FEAT, dtype=np.float32)
    x[r1 * GRID + c1] = 1.0
    x[GRID * GRID + r2 * GRID + c2] = 1.0
    return x


class QNet(nn.Module):
    """Shared MLP shape for both the per-agent and the joint-action critics."""

    def __init__(self, n_in, n_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


class Replay:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch, rng):
        mb = rng.sample(self.buf, batch)
        s, a, r, s2, d = zip(*mb)
        return (
            torch.from_numpy(np.stack(s)),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.from_numpy(np.stack(s2)),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def dqn_step(online, target, optimizer, replay, batch, gamma, rng):
    if len(replay) < batch:
        return
    s, a, r, s2, d = replay.sample(batch, rng)
    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next = target(s2).max(dim=1).values
        y = r + gamma * q_next * (1.0 - d)
    loss = nn.functional.mse_loss(q, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def epsilon_at(ep, decay_eps=1500, eps_start=1.0, eps_end=0.05):
    if ep >= decay_eps:
        return eps_end
    return eps_start + (eps_end - eps_start) * (ep / decay_eps)


# ----------------------------- Independent DQN -----------------------------

def independent_dqn(episodes=3000, alpha=1e-3, gamma=0.95, batch=64,
                    sync_every=500, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    nets = [QNet(N_FEAT, N_ACTIONS) for _ in range(2)]
    targets = [QNet(N_FEAT, N_ACTIONS) for _ in range(2)]
    for n, t in zip(nets, targets):
        t.load_state_dict(n.state_dict())
    opts = [torch.optim.Adam(n.parameters(), lr=alpha) for n in nets]
    replays = [Replay() for _ in range(2)]

    returns_log = []
    gstep = 0
    for ep in range(episodes):
        s = reset()
        eps = epsilon_at(ep)
        total = 0.0
        for _ in range(100):
            xs = torch.from_numpy(features(s)).unsqueeze(0)
            acts = []
            for i in range(2):
                if rng.random() < eps:
                    acts.append(rng.randrange(N_ACTIONS))
                else:
                    with torch.no_grad():
                        acts.append(int(nets[i](xs).argmax().item()))
            s_next, r, done = step(s, tuple(acts))
            total += r
            fx, fx2 = features(s), features(s_next)
            for i in range(2):
                replays[i].push(fx, acts[i], r, fx2, float(done))
                dqn_step(nets[i], targets[i], opts[i], replays[i], batch, gamma, rng)
            gstep += 1
            if gstep % sync_every == 0:
                for n, t in zip(nets, targets):
                    t.load_state_dict(n.state_dict())
            if done:
                break
            s = s_next
        returns_log.append(total)
    return nets, returns_log


# ----------------------------- Joint-action DQN -----------------------------

def joint_dqn(episodes=3000, alpha=1e-3, gamma=0.95, batch=64,
              sync_every=500, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = QNet(N_FEAT, N_JOINT)
    target = QNet(N_FEAT, N_JOINT)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=alpha)
    replay = Replay()

    returns_log = []
    gstep = 0
    for ep in range(episodes):
        s = reset()
        eps = epsilon_at(ep)
        total = 0.0
        for _ in range(100):
            xs = torch.from_numpy(features(s)).unsqueeze(0)
            if rng.random() < eps:
                ja_idx = rng.randrange(N_JOINT)
            else:
                with torch.no_grad():
                    ja_idx = int(net(xs).argmax().item())
            s_next, r, done = step(s, JOINT_ACTIONS[ja_idx])
            total += r
            replay.push(features(s), ja_idx, r, features(s_next), float(done))
            dqn_step(net, target, opt, replay, batch, gamma, rng)
            gstep += 1
            if gstep % sync_every == 0:
                target.load_state_dict(net.state_dict())
            if done:
                break
            s = s_next
        returns_log.append(total)
    return net, returns_log


# ----------------------------- eval / reporting -----------------------------

def block_mean(xs, block):
    return [sum(xs[i : i + block]) / block for i in range(0, len(xs) - block + 1, block)]


def evaluate_ind(nets, episodes=100):
    total = 0.0
    for _ in range(episodes):
        s = reset()
        for _ in range(100):
            xs = torch.from_numpy(features(s)).unsqueeze(0)
            with torch.no_grad():
                acts = tuple(int(nets[i](xs).argmax().item()) for i in range(2))
            s, r, done = step(s, acts)
            total += r
            if done:
                break
    return total / episodes


def evaluate_joint(net, episodes=100):
    total = 0.0
    for _ in range(episodes):
        s = reset()
        for _ in range(100):
            xs = torch.from_numpy(features(s)).unsqueeze(0)
            with torch.no_grad():
                ja_idx = int(net(xs).argmax().item())
            s, r, done = step(s, JOINT_ACTIONS[ja_idx])
            total += r
            if done:
                break
    return total / episodes


def main():
    print(f"=== Cooperative 2-agent GridWorld ({GRID}x{GRID}, shared reward) — deep RL ===")
    print(f"agents start at (0,0) and ({GRID-1}, 0); must both reach {GOAL}")
    print()

    episodes = 3000
    nets, log_ind = independent_dqn(episodes=episodes, seed=1)
    net_joint, log_joint = joint_dqn(episodes=episodes, seed=1)

    print("learning curves (mean return per 300 episodes):")
    for i, (a, b) in enumerate(zip(block_mean(log_ind, 300), block_mean(log_joint, 300))):
        print(f"  block {i+1}: independent-DQN = {a:7.2f}   joint-DQN = {b:7.2f}")

    print()
    print("final greedy evaluation (100 eps):")
    print(f"  independent-DQN mean return = {evaluate_ind(nets):.2f}")
    print(f"  joint-DQN       mean return = {evaluate_joint(net_joint):.2f}")

    print()
    print("note: joint-DQN factors the global view correctly, but its OUTPUT layer is |A|^2")
    print("(16 here; |A|^n for n agents). CTDE methods (MAPPO, QMIX) keep decentralized")
    print("actors but train a centralized critic to sidestep that exponential blow-up.")


if __name__ == "__main__":
    main()
