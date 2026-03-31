# AlphaZero Experiments

Unified AlphaZero implementation with a plug-and-play domain interface, applied to three games from Robin Langer's repos. Fixed the structural issues in the original implementations.

## Inspiration

This repo was inspired by — and fixes issues in — Robin's existing game implementations:
- [RaggedR/noriega](https://github.com/RaggedR/noriega) — Nonaga board game
- [RaggedR/blokus](https://github.com/RaggedR/blokus) — Blokus
- [RaggedR/checkers](https://github.com/RaggedR/checkers) — Checkers

The core finding that motivated this work: AlphaZero's self-play training signal degrades severely in games with extreme branching factors or draw-collapsed outcome distributions (Nonaga being the clearest example, where GAs outperformed AlphaZero 50-0 due to structural mismatch, not implementation bugs). This repo fixes the implementation and makes the structural requirements explicit.

## Architecture

The implementation uses a single `Domain` interface so any game can plug in without touching the core AlphaZero logic:

```
alphazero/
  core/       — MCTS, neural network, self-play loop
  domains/    — plug-and-play game implementations
    noriega/
    blokus/
    checkers/
  experiments/ — experiment scripts and results
```

### Domain Interface

Each domain implements:
```python
class Domain:
    def get_initial_state(self) -> State
    def get_legal_actions(self, state: State) -> List[Action]
    def apply_action(self, state: State, action: Action) -> State
    def is_terminal(self, state: State) -> bool
    def get_reward(self, state: State) -> float          # from current player's perspective
    def encode_state(self, state: State) -> np.ndarray   # for neural network input
    def action_space_size(self) -> int
```

## Workflow

- **Claudius** runs experiments and pushes to branches
- **Lyra** reviews and merges PRs

## Status

- [ ] Core AlphaZero implementation (MCTS + neural network)
- [ ] Noriega domain
- [ ] Blokus domain
- [ ] Checkers domain (pending Robin pushing the repo)
- [ ] Experiment results

## Language

Python (primary). Open to Haskell or Rust for performance-critical components if needed — the domain interface is language-agnostic.
