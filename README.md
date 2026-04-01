# AlphaZero Experiments

Unified, plug-and-play AlphaZero implementation applied to three games from
Robin Langer's repositories. A single AlphaZero core (MCTS + ResNet) works
with any game by implementing a clean `Game` interface — no game-specific
logic in the training loop.

## Inspiration and background

This repo was built alongside Robin's game implementations:

- **[RaggedR/nonaga](https://github.com/RaggedR/nonaga)** — Nonaga board game.
  Robin's repo documents 13 training attempts and a remarkable negative result:
  a 14-weight genetic algorithm beats a 570K-parameter AlphaZero neural network
  50-0. The root cause is structural: Nonaga games collapse to draws under
  competent self-play, starving the value head of signal. The AlphaZero code
  there is correct; the game resists standard self-play training.

- **[RaggedR/checkers](https://github.com/RaggedR/checkers)** — Checkers with
  TD(lambda) self-play (Arthur Samuel style). The AI in that repo uses a
  hand-tuned 10-feature linear evaluator trained via temporal difference
  learning — not AlphaZero. This repo wraps the same game engine for neural
  self-play training, which is a fair comparison point.

- **RaggedR/blokus** (referenced in MCTS_Laboratory PR #107, linked from the
  Nonaga README) — Robin found the same GA-beats-MCTS result in Blokus: an
  island-model GA with 10 heuristic weights beat FastMCTS 8-to-1. Same cause:
  high branching factor (~80-500) makes random rollouts uninformative. AlphaZero
  (no random rollouts) is the right neural approach to test against.

**What this repo adds:** a shared, correct AlphaZero implementation with a
documented `Game` interface, so the same training loop can be applied to all
three games without copy-pasting MCTS or network code.

## Architecture

```
alpha_zero/
  game.py        Abstract Game interface (8 methods)
  mcts.py        Monte Carlo Tree Search with PUCT selection
  network.py     ResNet with policy + value heads (game-agnostic)
  trainer.py     Self-play → train → arena loop
  utils.py       Logging, timers, helpers
  __init__.py

games/
  nonaga.py      Nonaga adapter (two-ply turns: piece move + tile move)
  checkers.py    Checkers adapter (32-square board, mandatory captures)
  blokus.py      Blokus adapter (2-player, 20×20, polyomino placement)
  __init__.py

main.py          Entry point: python main.py --game <name>
requirements.txt torch, numpy
```

### The Game interface

Every game adapter implements these eight methods:

```python
class Game(ABC):
    def get_init_state(self) -> State
    def get_next_state(self, state, player, action) -> (next_state, next_player)
    def get_valid_moves(self, state, player) -> List[action]
    def get_game_ended(self, state, player) -> float  # 0=ongoing, 1=win, -1=loss, 1e-4=draw
    def get_canonical_form(self, state, player) -> State
    def get_action_size(self) -> int
    def get_state_representation(self, state) -> np.ndarray  # (C, H, W) for neural net
    def get_display_name(self) -> str
```

Optional overrides for efficiency: `action_to_index`, `index_to_action`,
`get_valid_moves_mask`, `get_symmetries`.

The `get_canonical_form` contract: always return the board from the perspective
of player +1. The network always predicts for "the player about to move". This
keeps the training loop symmetric and avoids sign-confusion bugs (one of the
instabilities documented in RaggedR/checkers).

### MCTS

`alpha_zero/mcts.py` implements AlphaZero MCTS with:
- **PUCT selection** — Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))
- **Neural network priors** — policy head guides exploration; value head
  replaces random rollouts entirely.
- **Dirichlet noise at root** — configurable alpha and epsilon.
- **Correct value backup for two-ply games** — sign flips only at player
  boundaries, not at every tree level. The Nonaga repo documents the sign-
  confusion bug that arises when this is done naively; this implementation
  tracks player identity at each node.

### Network

`alpha_zero/network.py` — game-agnostic ResNet:
- Input shape determined at construction time from
  `game.get_state_representation(game.get_init_state()).shape`.
- Policy head outputs logits over the full action space (illegal actions
  masked by MCTS).
- Value head outputs a scalar in [-1, 1] via Tanh.
- Default: 6 residual blocks, 128 channels (~3M parameters).

### Trainer

`alpha_zero/trainer.py` — standard AlphaZero loop:
1. **Self-play** — generate examples (state, MCTS policy, outcome).
2. **Train** — policy cross-entropy + value MSE on replay buffer.
3. **Arena** — new vs old network; keep new if win rate ≥ threshold (default 55%).

## Usage

```bash
pip install -r requirements.txt

# Train on checkers (easiest — well-defined outcomes, moderate branching)
python main.py --game checkers

# Train on Nonaga (hard — expect poor results, as Robin documented)
python main.py --game nonaga --sims 50 --iterations 30

# Train on Blokus 2-player
python main.py --game blokus --channels 64 --sims 50

# Larger run with custom settings
python main.py --game checkers \
    --iterations 200 \
    --games 200 \
    --sims 400 \
    --channels 256 \
    --res-blocks 10 \
    --checkpoint-dir checkpoints

# Evaluate a saved checkpoint vs random
python main.py --game checkers --load checkpoints/checkers/best.pt --eval-only
```

## Known structural difficulties

| Game | Branching | Draw rate | AlphaZero outlook |
|------|-----------|-----------|-------------------|
| Checkers | ~7 | Low | Good — decisive outcomes, tractable branching |
| Nonaga | ~300 (two-ply) | Very high | Hard — value signal collapses; GAs dominate |
| Blokus (2p) | ~80-500 | Low | Moderate — branching is the challenge |

These observations come directly from Robin's experimental writeups. The
implementation here is correct for all three; the structural difficulty is a
property of the games, not the code.

## Workflow

- **Claudius** runs experiments and opens PRs with results.
- **Lyra** reviews, suggests hyperparameter changes, and merges.

## References

- Silver et al. (2017). "Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm." arXiv:1712.01815.
- RaggedR/nonaga `DESIGN_DECISIONS.md` — 13-attempt training history for Nonaga.
- Samuel (1959). "Some Studies in Machine Learning Using the Game of Checkers."
  IBM Journal of Research and Development.
- MCTS_Laboratory PR #107 — island-model GA vs FastMCTS in Blokus.
