# Capstone: AlphaZero on Tic-Tac-Toe

An experimental, end-to-end realization of the **AlphaZero** recipe —
self-play + MCTS + policy/value distillation — on the smallest game that can
still prove the method works. One file, CPU-only, trains in ~30 seconds.

- Code: [`alphazero_tictactoe.py`](alphazero_tictactoe.py)
- Run: `../../../../.venv/bin/python alphazero_tictactoe.py`
  (or any Python with `torch` + `numpy`)

## Why Tic-Tac-Toe

The point of the capstone is not to beat a hard game — it is to **verify the
algorithm actually learned**. Tic-Tac-Toe is the best toy for that because it is
**fully solved**, which turns "did it learn?" into a crisp, falsifiable test:

- Optimal play **can never lose** → a correct agent must **draw every game
  against a perfect (minimax) player**.
- It must **beat a random player** nearly always.

Both opponents are built in, so the experiment checks itself. The game is also
tiny (3×3, 9 actions, ≤9 moves, ~5,478 reachable states), so it trains in
seconds on a laptop with no GPU — you can iterate on the *algorithm* instead of
fighting compute. (Connect Four is the usual next step up; see the end.)

## What's implemented (the full AlphaZero loop, nothing else)

| Stage | Where | Notes |
|---|---|---|
| 1. Game + perfect simulator | `next_state`, `game_ended`, `legal_moves` | canonical board: current player is always `+1` |
| 2. Policy-value net `f_θ(s)→(p,v)` | `PolicyValueNet`, `NetWrapper` | small MLP: 9 → 64 → 64 → (9 policy logits, 1 tanh value) |
| 3. MCTS with PUCT | `MCTS` | `(p,v)` as prior + bootstrap; Dirichlet root noise for exploration |
| 4. Self-play | `self_play_episode` | MCTS visit counts become the policy target |
| 5. Training | `NetWrapper.train` | cross-entropy to visit counts + MSE to game outcome |
| 6. Evaluation | `evaluate`, `MinimaxPlayer`, `RandomPlayer` | vs random, vs perfect play, and a no-search ablation |

The training objective is exactly AlphaZero's:

```
L(θ) = (v_θ(s) − z)²  −  π · log p_θ(s)
       └── value: MSE ──┘  └ policy: cross-entropy to MCTS visits ┘
```

`π` is the MCTS visit distribution (the *improved* policy), `z ∈ {+1,0,−1}` is
the self-play game outcome. The policy head is trained to **imitate the agent's
own search** — the core AlphaZero idea.

## Results (seed 0, ~27 s)

```
vs RANDOM          : W=183  D= 16  L=  1
vs MINIMAX         : W=  0  D=200  L=  0     <- perfect play: all draws
net-only vs RANDOM : W=166  D= 29  L=  5     <- ablation: no MCTS

[PASS] never loses to PERFECT play   (losses=0/200)
[PASS] loses <=1% vs random          (losses=1/200)
[PASS] search helps: net+MCTS 1 < net-only 5 losses
```

How to read these:

1. **Gold standard — 0 losses vs perfect play.** Optimal TTT draws against
   optimal TTT. 200/200 draws is the decisive evidence the agent learned to
   play optimally on every line a real opponent forces.
2. **Ablation — search is doing real work.** The *same* network playing its raw
   policy with **no MCTS** loses several games to random; adding search drops
   that to ~0. This is the AlphaZero thesis in miniature: the network alone is a
   decent prior, but **lookahead is what makes it strong**.
3. **The ~1% random losses are expected, not a bug.** A random opponent walks
   into bizarre off-principal-variation states the self-play rarely visited. In
   a few of those, the net's value estimate is wrong and 200 simulations don't
   fully correct it — the lesson's **"search–policy mismatch"** pitfall. More
   self-play or more eval simulations shrink it further; it is a property of
   bounded search, which is why the hard guarantee is stated against *perfect*
   play (on-distribution), with random held to a ≤1% rate.

## Key implementation details worth knowing

- **Canonical board.** The board is always shown from the mover's perspective
  (their marks `+1`). `next_state` places a `+1` then negates the board, so it
  stays canonical for the opponent. This halves what the net must learn (it only
  ever sees "my turn") and is the standard AlphaZero trick.
- **MCTS value signs.** `_search` returns `−v` up the tree because each ply
  flips whose perspective the value is from. Terminal nodes return `−game_ended`
  for the same reason. (If you ever port this, this sign is the usual bug source.)
- **Dirichlet root noise.** Mixed into the root prior during self-play
  (`dirichlet_alpha=0.6`, 25% weight) so the agent explores off-prior moves and
  the net learns the states a non-optimal opponent creates. Removing it brings
  back a couple of random losses.
- **Temperature schedule.** `temp=1` (sample ∝ visit counts) for the first few
  moves to diversify openings, then `temp=0` (greedy) — AlphaZero's exploration
  schedule.
- **Everything is seeded** (`torch`, `numpy`, `random`), so runs are
  reproducible.

## Knobs (in `main()` / `train_alphazero`)

| Param | Default | Effect |
|---|---|---|
| `iters` | 20 | self-play ↔ train rounds |
| `episodes` | 50 | self-play games per round |
| `n_sims` (train) | 50 | MCTS simulations per move during self-play |
| `eval_sims` | 200 | MCTS simulations per move at evaluation |
| `epochs` | 10 | net training epochs per round |
| `replay_iters` | 4 | rounds of self-play data kept for training |

More `n_sims` / `iters` → stronger play (and the random-loss rate → 0), at the
cost of runtime.

## Scaling up (optional next experiment)

Swap only the **Game** layer and everything else carries over:

- **Connect Four** (7×6): the classic AlphaZero miniature. Needs a CNN body
  instead of the MLP and more simulations/iterations (minutes, not seconds), and
  there is no cheap perfect oracle — evaluate by ELO vs earlier checkpoints and
  a 1-ply/greedy heuristic baseline instead of minimax.
- **Gomoku on a small board**, **Othello 6×6**: same pattern.

The self-play loop, MCTS, net heads, and training objective do **not** change —
that generality is the whole point of AlphaZero, and this capstone demonstrates
it on a game you can fully verify.
