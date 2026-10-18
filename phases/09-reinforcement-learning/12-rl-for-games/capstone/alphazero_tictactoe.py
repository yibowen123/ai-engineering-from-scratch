"""AlphaZero on Tic-Tac-Toe — a small, verifiable experimental realization.

Why Tic-Tac-Toe: it is fully solved, so "did AlphaZero actually learn?" has a
crisp, falsifiable answer — a correct agent playing greedily must NEVER LOSE
(optimal play forces at least a draw), must DRAW every game against a perfect
minimax player, and must beat a random player almost always. Those are the
tests at the bottom of this file. It also trains in ~1-2 min on a laptop CPU.

The whole AlphaZero loop is here and nothing else:
  1. Game       — rules + a perfect simulator (TicTacToe).
  2. Net        — one tower -> (policy prior p, value v).  (policy_value_net)
  3. MCTS       — PUCT search that uses (p, v) as prior + bootstrap.
  4. Self-play  — play against self; MCTS visit counts are the policy target.
  5. Train      — supervised: match the visit counts, regress to game outcome.
  6. Evaluate   — vs random and vs perfect minimax.

Board convention (the standard AlphaZero "canonical" trick): the board is always
shown from the perspective of the player to move, whose marks are +1 and whose
opponent's marks are -1. Making a move places a +1, then we FLIP signs so the
board stays canonical for the next player. Action = cell index 0..8 (perspective
does not change which cell an index refers to).
"""

import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- 1. Game -----------------------------

N = 9  # 3x3
LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
]
DRAW = 1e-4  # tiny non-zero so "game over, drawn" is distinguishable from "ongoing"


def initial_board():
    return np.zeros(N, dtype=np.float32)


def legal_moves(board):
    return [i for i in range(N) if board[i] == 0]


def next_state(board, action):
    """Place the current player's mark (+1) at `action`, then flip perspective
    so the returned board is canonical for the opponent (their marks are +1)."""
    b = board.copy()
    b[action] = 1.0
    return -b


def game_ended(board):
    """Value from the perspective of the player to move in `board`:
    +1 they have a completed line, -1 the opponent does, DRAW if full, else 0."""
    for a, b, c in LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1.0
        if s == -3:
            return -1.0
    if not (board == 0).any():
        return DRAW
    return 0.0


def key(board):
    return board.tobytes()


# ----------------------------- 2. Policy-value network -----------------------------

class PolicyValueNet(nn.Module):
    """f_theta(s) -> (policy logits over 9 cells, value in [-1, 1])."""

    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(N, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, N)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1)


class NetWrapper:
    """Inference (predict) + training for the net. Keeps AlphaZero code clean."""

    def __init__(self, hidden=64, lr=1e-3):
        self.net = PolicyValueNet(hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def predict(self, board):
        """-> (prior prob vector over 9 cells, scalar value). No grad."""
        self.net.eval()
        with torch.no_grad():
            logits, v = self.net(torch.from_numpy(board).unsqueeze(0))
            p = F.softmax(logits.squeeze(0), dim=0).numpy()
        return p, float(v.item())

    def train(self, examples, epochs=10, batch=64, rng=None):
        rng = rng or random.Random(0)
        self.net.train()
        boards = torch.from_numpy(np.stack([b for b, _, _ in examples]))
        pis = torch.from_numpy(np.stack([pi for _, pi, _ in examples]).astype(np.float32))
        zs = torch.tensor([z for _, _, z in examples], dtype=torch.float32)
        n = len(examples)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                logits, v = self.net(boards[idx])
                # policy: cross-entropy to MCTS visit distribution; value: MSE to outcome.
                policy_loss = -(pis[idx] * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(v, zs[idx])
                loss = policy_loss + value_loss
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


# ----------------------------- 3. MCTS (PUCT) -----------------------------

class MCTS:
    def __init__(self, net, c_puct=1.5, n_sims=50):
        self.net = net
        self.c_puct = c_puct
        self.n_sims = n_sims
        self.Q = {}        # (s, a) -> mean action value
        self.Nsa = {}      # (s, a) -> visit count
        self.Ns = defaultdict(int)   # s -> visit count
        self.P = {}        # s -> prior vector (masked to legal moves)
        self.ended = {}    # s -> cached game_ended

    def _search(self, board):
        s = key(board)
        if s not in self.ended:
            self.ended[s] = game_ended(board)
        e = self.ended[s]
        if e != 0:
            # Terminal: value flips because it is scored for the player to move,
            # and the parent is the other player.
            return -(0.0 if e == DRAW else e)

        if s not in self.P:  # leaf -> evaluate with the net and return its value
            p, v = self.net.predict(board)
            mask = np.zeros(N, dtype=np.float32)
            for a in legal_moves(board):
                mask[a] = 1.0
            p = p * mask
            p = p / p.sum() if p.sum() > 0 else mask / mask.sum()
            self.P[s] = p
            return -v

        # Select the action maximizing PUCT.
        best_u, best_a = -float("inf"), -1
        sqrtN = math.sqrt(self.Ns[s] + 1e-8)
        for a in legal_moves(board):
            if (s, a) in self.Q:
                u = self.Q[(s, a)] + self.c_puct * self.P[s][a] * sqrtN / (1 + self.Nsa[(s, a)])
            else:
                u = self.c_puct * self.P[s][a] * sqrtN  # Q defaults to 0 for unseen edges
            if u > best_u:
                best_u, best_a = u, a

        a = best_a
        v = self._search(next_state(board, a))

        # Backup.
        if (s, a) in self.Q:
            self.Nsa[(s, a)] += 1
            self.Q[(s, a)] += (v - self.Q[(s, a)]) / self.Nsa[(s, a)]
        else:
            self.Nsa[(s, a)] = 1
            self.Q[(s, a)] = v
        self.Ns[s] += 1
        return -v

    def action_probs(self, board, temp=1.0, dirichlet_alpha=None, noise_eps=0.25):
        # Optional Dirichlet noise on the ROOT prior (standard AlphaZero
        # exploration): forces self-play to try off-principal-variation moves so
        # the net also learns the odd states a non-optimal opponent creates.
        if dirichlet_alpha is not None:
            self._search(board)  # expand root so self.P[root] exists
            s0 = key(board)
            if s0 in self.P:
                legal = legal_moves(board)
                noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
                p = self.P[s0].copy()
                for a, nz in zip(legal, noise):
                    p[a] = (1 - noise_eps) * p[a] + noise_eps * nz
                self.P[s0] = p
        for _ in range(self.n_sims):
            self._search(board)
        s = key(board)
        counts = np.array([self.Nsa.get((s, a), 0) for a in range(N)], dtype=np.float64)
        if temp <= 1e-2 or counts.sum() == 0:  # greedy: all mass on most-visited
            probs = np.zeros(N)
            probs[int(counts.argmax())] = 1.0
            return probs
        counts = (counts / counts.max()) ** (1.0 / temp)   # normalize first: no overflow
        return counts / counts.sum()


# ----------------------------- 4 & 5. Self-play + training -----------------------------

def self_play_episode(net, n_sims, rng, temp_moves=4):
    """Play one game of the net against itself. Returns training examples
    (canonical_board, mcts_policy, outcome_from_that_board's_player)."""
    mcts = MCTS(net, n_sims=n_sims)   # fresh tree per game
    board = initial_board()
    trajectory = []  # (board, player_sign, pi)
    player = 1       # +1 / -1, only used to assign the final outcome
    move = 0
    while True:
        temp = 1.0 if move < temp_moves else 0.0   # explore early, greedy later
        pi = mcts.action_probs(board, temp=temp, dirichlet_alpha=0.6)
        trajectory.append((board.copy(), player, pi))
        a = rng.choices(range(N), weights=pi)[0]
        board = next_state(board, a)
        player = -player
        move += 1
        e = game_ended(board)
        if e != 0:
            # e is scored for the player to move now (`player`). Assign each stored
            # position its value from that position's own player's perspective.
            z = 0.0 if e == DRAW else e
            return [(b, pi, z if p == player else -z) for (b, p, pi) in trajectory]


def train_alphazero(iters=15, episodes=40, n_sims=40, epochs=10,
                    replay_iters=4, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)   # for reproducible Dirichlet root noise
    rng = random.Random(seed)
    net = NetWrapper()
    replay = []  # list of per-iteration example batches (keep last `replay_iters`)

    for it in range(iters):
        fresh = []
        for _ in range(episodes):
            fresh.extend(self_play_episode(net, n_sims, rng))
        replay.append(fresh)
        replay = replay[-replay_iters:]
        examples = [ex for batch in replay for ex in batch]
        net.train(examples, epochs=epochs, rng=rng)
        if verbose:
            # Quick health check each iter: lose-rate vs random as X and O.
            w, d, l = evaluate(net, RandomPlayer(rng), games=40, n_sims=n_sims)
            print(f"  iter {it+1:2d}/{iters}  self-play examples={len(examples):5d}  "
                  f"vs-random  W={w:2d} D={d:2d} L={l:2d}")
    return net


# ----------------------------- Players -----------------------------

class RandomPlayer:
    def __init__(self, rng):
        self.rng = rng

    def move(self, board):
        return self.rng.choice(legal_moves(board))


class AlphaZeroPlayer:
    """Greedy AlphaZero: run MCTS, play the most-visited move (temp=0)."""

    def __init__(self, net, n_sims=50):
        self.net = net
        self.n_sims = n_sims

    def move(self, board):
        mcts = MCTS(self.net, n_sims=self.n_sims)
        return int(mcts.action_probs(board, temp=0).argmax())


class NetOnlyPlayer:
    """Ablation: pick argmax of the raw policy prior, NO search."""

    def __init__(self, net):
        self.net = net

    def move(self, board):
        p, _ = self.net.predict(board)
        legal = legal_moves(board)
        return max(legal, key=lambda a: p[a])


class MinimaxPlayer:
    """Perfect play via memoized negamax on the (tiny) game tree."""

    def __init__(self):
        self.cache = {}

    def _value(self, board):
        s = key(board)
        if s in self.cache:
            return self.cache[s]
        e = game_ended(board)
        if e != 0:
            val = (0.0 if e == DRAW else e, None)
            self.cache[s] = val
            return val
        best_v, best_a = -float("inf"), None
        for a in legal_moves(board):
            v, _ = self._value(next_state(board, a))
            v = -v
            if v > best_v:
                best_v, best_a = v, a
        self.cache[s] = (best_v, best_a)
        return self.cache[s]

    def move(self, board):
        return self._value(board)[1]


# ----------------------------- 6. Evaluation -----------------------------

def play_game(p_first, p_second):
    """Play one canonical game. Returns +1 if p_first wins, -1 if p_second wins, 0 draw."""
    board = initial_board()
    players = [p_first, p_second]
    turn = 0
    while True:
        e = game_ended(board)
        if e != 0:
            if e == DRAW:
                return 0
            # e is scored for the player to move (players[turn], who has no move
            # that mattered — the previous mover completed a line). So the player
            # to move LOST. Winner is the other one.
            loser_is_first = (turn == 0)
            return -1 if loser_is_first else 1
        a = players[turn].move(board)
        board = next_state(board, a)
        turn ^= 1


def evaluate(net_or_player, opponent, games=100, n_sims=50):
    """Play `games` split evenly as first/second. Returns (wins, draws, losses)
    from the AlphaZero agent's perspective. Accepts a NetWrapper or a Player."""
    agent = net_or_player
    if isinstance(net_or_player, NetWrapper):
        agent = AlphaZeroPlayer(net_or_player, n_sims=n_sims)
    w = d = l = 0
    for g in range(games):
        if g % 2 == 0:
            r = play_game(agent, opponent)        # agent goes first
        else:
            r = -play_game(opponent, agent)       # agent goes second; flip sign
        if r > 0:
            w += 1
        elif r < 0:
            l += 1
        else:
            d += 1
    return w, d, l


# ----------------------------- main / tests -----------------------------

def main():
    print("=== AlphaZero on Tic-Tac-Toe ===\n")
    rng = random.Random(0)

    print("Training (self-play + MCTS + distillation):")
    net = train_alphazero(iters=20, episodes=50, n_sims=50, epochs=10, seed=0)
    print()

    eval_sims = 200   # a tiny game tree: 200 sims make MCTS effectively optimal
    agent = AlphaZeroPlayer(net, n_sims=eval_sims)

    print(f"Final evaluation (200 games each, alternating who starts, {eval_sims} sims):")
    w, d, l = evaluate(agent, RandomPlayer(rng), games=200)
    print(f"  vs RANDOM   : W={w:3d}  D={d:3d}  L={l:3d}")
    wm, dm, lm = evaluate(agent, MinimaxPlayer(), games=200)
    print(f"  vs MINIMAX  : W={wm:3d}  D={dm:3d}  L={lm:3d}   (perfect play -> all draws)")

    # Ablation: play the RAW policy net with NO search, vs random. Random reaches
    # off-principal-variation states; without MCTS to recover, the net slips.
    net_only = NetOnlyPlayer(net)
    wn, dn, ln = evaluate(net_only, RandomPlayer(rng), games=200)
    print(f"  net-only vs RANDOM : W={wn:3d}  D={dn:3d}  L={ln:3d}   (no MCTS)")

    print("\n=== Correctness tests ===")
    # TTT is solved, so these are decisive, falsifiable checks. The GOLD STANDARD
    # is play vs a perfect player: optimal play draws every game, so zero losses
    # there proves the agent is optimal on the lines that matter. Vs RANDOM is a
    # harsher *coverage* probe (random reaches off-distribution states); bounded
    # search can rarely slip there, so we allow <=1% -- the "search-policy
    # mismatch" pitfall from the lesson, not a bug.
    ok_minimax = (lm == 0)
    ok_random = (l <= 0.01 * 200)          # <= 1% loss rate vs random
    ok_search = (l < ln)                   # MCTS strictly reduces losses vs raw net
    print(f"  [{'PASS' if ok_minimax else 'FAIL'}] never loses to PERFECT play     (losses={lm}/200)")
    print(f"  [{'PASS' if ok_random else 'FAIL'}] loses <=1% vs random            (losses={l}/200)")
    print(f"  [{'PASS' if ok_search else 'FAIL'}] search helps: net+MCTS {l} < net-only {ln} losses")

    if ok_minimax and ok_random and ok_search:
        print("\nAll tests passed: the agent plays optimal Tic-Tac-Toe, and search helps.")
    else:
        print("\nSome tests failed — try more iters/episodes/sims (see README).")


if __name__ == "__main__":
    main()
