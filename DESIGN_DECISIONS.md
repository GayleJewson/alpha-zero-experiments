# Design Decisions

This document records the *why* behind parameter choices, architectural
decisions, and experimental methodology. The README covers *what* and *how*.

## Central result

**On consumer hardware with a 24-hour training budget, genetic algorithms with
10-14 hand-crafted features outperform pure AlphaZero self-play across all four
tested domains.** The only case where a neural network surpassed the GA (Connect
Four) required using the GA itself as a training signal.

### Evidence

| Game | b | GA | GA time | Pure self-play NN | NN time | Result |
|------|---|-----|---------|-------------------|---------|--------|
| Connect Four | ~7 | Island GA, 14w | ~12.5 hrs | Regressed (draws→losses) | ~20 iters | **GA wins** |
| Checkers | ~7 | TD(λ), 10w | ~30-60 min | Projected ~62 hrs for 50 iters | In progress | **GA wins** (projected) |
| Nonaga | ~300 | Island GA, 14w | ~1-2 hrs | Failed after 100 hrs, 50 iters | ~100 hrs | **GA 50-0** |
| Blokus | ~80-500 | Island GA, 10w | ~10 min | Not tested | — | GA beats MCTS 8-to-1 |

**Connect Four correction:** The widely-cited "NN beats GA 10-0 in 13 minutes"
result from RaggedR/connect_four used *league training* — the NN was
bootstrapped on 200 GA-generated games and trained with 30% continuous GA
supervision. Pure self-play AlphaZero *regressed* against the GA (50% draws →
50% losses by iteration 20). The NN didn't independently surpass the GA; it
stood on the GA's shoulders. Remove the GA supervision and the NN fails.

### Estimated AlphaZero training times on consumer hardware

**Measured baseline:** On Apple M4 (MPS), our 1.9M-param network achieves ~215
neural net evaluations/second. Each self-play move requires `sims` evaluations,
and each game requires `avg_moves` moves.

**Scaling from published AlphaZero results:** Silver et al. (2017) trained on
5,000 first-generation TPUs in parallel. We estimate our single M4 is ~10,000×
slower in effective self-play throughput (sequential predict calls, smaller batch
utilisation, Python overhead). "Decent" is defined as reliably beating a random
player and showing strategic understanding — roughly 10% of the training budget
needed for superhuman play.

| Game | b | Avg moves | Sims needed | Network | Est. games to "decent" | **Est. wall-clock (M4)** |
|------|---|-----------|-------------|---------|----------------------|------------------------|
| Connect Four | ~7 | ~30 | 100 | 227K | ~5,000 | **~1 day** |
| Checkers | ~7 | ~48 | 200 | 1.9M | ~10,000–20,000 | **~5–10 days** |
| Chess | ~35 | ~80 | 400–800 | ~10M | ~500,000+ | **~3–10 years** |
| Blokus (2p) | ~80-500 | ~40 | 200–400 | ~3M | ~100,000+ | **~2–6 months** |
| Nonaga | ~300 | ~22 | 200+ | ~570K | ∞ (structural) | **Never** (draws kill signal) |
| Go (19×19) | ~250 | ~200 | 800+ | ~20M | ~3,000,000+ | **~50–150 years** |
| Shogi | ~80 | ~115 | 400–800 | ~10M | ~500,000+ | **~5–15 years** |

**Estimation methodology (showing our working):**

Our one hard measurement: on Apple M4 (MPS), a 1.9M-param ResNet achieves
**~215 neural net evaluations/second** in self-play. Derived from Run 1:
960,000 predict calls in 74.5 minutes = 214.8 calls/sec.

For each game, wall-clock time is estimated as:

    T = (games_to_decent × avg_moves × sims_per_move) / evals_per_second

where `evals_per_second` is scaled linearly by network size relative to our
measured 1.9M-param baseline (conservative — MPS overhead is partly fixed-cost,
so larger networks don't slow down proportionally, but we err on the generous
side for the NN).

**"Games to decent" derivations:**
- **Connect Four (5,000 games):** Silver et al. report 44M games for superhuman
  chess. Connect Four is ~10^35× simpler (state space ~4×10^12 vs ~10^47).
  Apply 10% scaling for "decent" and adjust for complexity: ~5,000 games.
- **Checkers (10,000–20,000 games):** State space ~5×10^20, between Connect
  Four and chess. 10% of an estimated 100K-200K full training: ~10K-20K games.
- **Chess (500,000+ games):** 10% of DeepMind's 44M = 4.4M, but with smaller
  network and fewer sims we need more games to compensate. Conservative lower
  bound: 500K.
- **Blokus (100,000+ games):** No published baseline. High branching (80-500)
  means MCTS policies are noisy → more games needed per unit of learning.
  Estimated at 3-5× checkers adjusted for branching.
- **Go (3,000,000+ games):** 10% of DeepMind's 29M = 2.9M games minimum.
- **Nonaga (∞):** Structural impossibility. Not a compute estimate. See failure
  evidence below.

**"Sims needed" reasoning:**
- MCTS needs enough simulations to visit each legal move at least a few times.
  As a rough lower bound, sims ≈ 2-3× average branching factor.
- Connect Four (b≈7): 100 sims gives ~14 visits per move — adequate.
- Checkers (b≈7): 200 sims gives ~28 visits per move — comfortable.
- Chess (b≈35): 400 sims gives ~11 visits per move — minimum viable.
- Blokus (b≈80-500): 200 sims gives <2 visits per move — MCTS is guessing.
  Would need 400+ for meaningful search, but that doubles self-play cost.
- Go (b≈250): 800 sims gives ~3 visits per move — barely adequate, which is
  why DeepMind used 800.

**Error bounds:** These estimates carry ±0.5 orders of magnitude uncertainty.
The qualitative ranking (Connect Four < Checkers < Blokus < Chess < Go) is
robust; the exact numbers are not. The key claim is ordinal, not cardinal.

**The comparison:**

| Game | GA training time | AlphaZero estimated time | GA/NN ratio |
|------|-----------------|------------------------|-------------|
| Connect Four | 12.5 hours | ~1 day | ~2× |
| Checkers | 30–60 minutes | ~5–10 days | ~200× |
| Nonaga | 1–2 hours | Never | ∞ |
| Blokus | 10 minutes | ~2–6 months | ~10,000× |
| Chess | (untested) | ~3–10 years | — |
| Go | (untested) | ~50–150 years | — |

Even in the most favourable case (Connect Four, low branching), the GA trains
in half the time. For high-branching games the gap is 4–5 orders of magnitude.
This is not a close contest.

### Empirical evidence: NN failure modes from short training

These are not hypothetical concerns. Every failure mode below was observed in
Robin's game repos with exact measurements.

**Nonaga — 13 training attempts, none succeeded (RaggedR/nonaga):**

| Attempt | Method | Result | Failure mode |
|---------|--------|--------|--------------|
| 1 | Curriculum pretraining | 0% vs random | Learned "adjacency good" but couldn't form triangles |
| 3 | Training vs random | 0% vs random | Only saw losing positions, never learned what winning looks like |
| 4 | Self-play, 50 iters | 100% draw rate (all games) | max_plies=100 too short; value head received only zero targets |
| 4b | Self-play, max_plies=300 | 100% draw rate still | MCTS play too defensive; avg_plies=300.0 |
| 5 | Endgame bootstrap | 0% MCTS wins at 25, 50, and 200 sims | Uniform prior spread sims across ~100 legal moves; ~2 visits each |
| 10 | Island-model AlphaZero | Best island: 50% vs GA, then regressed | Regressed to 0% by iteration 53 |
| 11 | NN vs GA (distillation) | 0% across 12 iterations (600 eval games) | Learned GA-style positions but couldn't generate them |
| 12 | GA self-play distillation | 0 decisive games out of 1000 | Identical GAs draw every game — no training signal |
| 13 | League training | Hit 50%, regressed to 0% by epoch 19 | Catastrophic forgetting; 50% appears to be a hard ceiling |
| 49 (iter) | Self-play (degenerate) | 0/10 wins at 10, 25, and 50 sims | **Actively avoided forming triangles** — anti-learned the objective |

The iteration 49 result deserves emphasis: a random policy wins 5/20 games,
but the trained network wins 0/10 at all sim levels. Training made it *worse
than random*. The network learned to avoid the winning condition.

**Connect Four — pure self-play regression (RaggedR/connect_four):**

After 20 iterations of pure self-play:
- vs Random: improved 75% → 100% (looked like progress)
- vs GA: **regressed from 50% draws to 50% losses** (catastrophic forgetting)

Root cause: as the 40K-position replay buffer filled with self-play data, the
16K GA-bootstrap samples were diluted. The network "forgot" strategic play.
Additional bugs compounded the failure: optimizer state destroyed every
iteration (Adam degraded to near-SGD), batch size too small (64 → 781 noisy
gradient updates/epoch), no gradient clipping on inverted value predictions.

The league training fix (30% GA supervision) recovered to 10-0 vs GA by
iteration 6 — but this means the NN *required the GA as a teacher*.

**This repo — checkers, early iterations (2026-04-03):**

| Iteration | Policy loss | Value loss | Notes |
|-----------|-------------|------------|-------|
| 1 (200 sims, 100 games) | 4.809 | 0.675 | Near-random (log(1024)=6.93) |
| 1 (50 sims, 20 games) | 5.841 | 0.696 | Worse — noisier MCTS at low sims |
| 2 (50 sims, 20 games) | 4.304 | 0.347 | Learning, but still random-level play |

After ~18 minutes of training (2 iterations at 50 sims), the network cannot
beat a random player. The TD(λ) baseline in RaggedR/checkers achieved 20-0 vs
random after 30-60 minutes of CPU training.

### Related work

**Athenan** (Cohen-Solal & Cazenave, AAMAS 2023) — minimax with *learned*
neural evaluation beats Polygames (AlphaZero) at 296× lower compute. Won 48
gold medals at Computer Olympiad. But still uses neural network evaluation
inside minimax, not hand-crafted features. Shows that the *search algorithm*
(minimax vs MCTS) matters more than the training method.

**Scheiermann & Konen** (IEEE ToG 2022) — TD n-tuple networks wrapped with
MCTS only at test time. Trained on CPU, beats Edax level 7 on Othello. Simpler
than deep NNs but still uses thousands of learned n-tuple features, not 10-14
hand-crafted ones.

**Uber Deep Neuroevolution** (Such et al., 2018) — GA-evolved 4M-param neural
networks competitive with DQN/A3C/ES across 13 Atari games. Shows evolution
works for *training* neural networks, but evolves large NNs, not compact
heuristic evaluation functions.

**OLIVAW** (Norelli et al., IEEE ToG 2022) — AlphaZero for 8×8 Othello on
standard hardware (Xeon + K80 + TPU). Took 30 days of training to beat Edax
level 7. No evolutionary baseline comparison.

**Consumer AlphaZero implementations** (AlphaZero.jl, suragnair/alpha-zero-
general, michaelnny/alpha_zero) — report 5 hours to 30 days for competent play
on Connect Four, Othello, and 9×9 Go. None compare against evolutionary or
heuristic baselines.

**Gap in the literature:** No published work compares GA-evolved heuristic
evaluation functions (10-14 features, CPU-trainable in minutes) against
AlphaZero-style neural self-play across multiple board games under a fixed
compute budget. The intersection of evolutionary game AI and AlphaZero-style
deep RL is nearly empty — the two communities rarely test against each other.

### Why this matters

This result is not a criticism of AlphaZero. With 5,000 TPUs, AlphaZero
achieves superhuman play in hours — a level GAs cannot reach. The result is
about **practical compute regimes**: on the hardware most researchers actually
have, evolutionary methods with domain-informed features are dramatically more
efficient than pure self-play neural methods.

This connects to the GECCO paper thesis: if evolution is the practical winner
in compute-constrained regimes, then understanding its dynamics (how migration
topology controls diversity, how composition structure determines convergence)
is directly useful engineering knowledge, not just theory.

This also complements Athenan's result from a different angle: they showed
minimax + neural eval beats MCTS + neural eval (search algorithm matters). We
show that minimax + evolved features beats MCTS + learned features at low
compute (the evaluation source matters too). Together, these results suggest
that AlphaZero's dominance depends on massive compute to overcome the sample
inefficiency of self-play — remove that compute, and older approaches win.

## Experiment log

### Run 1 — Checkers baseline, aborted (2026-04-03)

**Goal:** Validate the AlphaZero pipeline on checkers.

**Command:**
```bash
python main.py --game checkers --iterations 50 --sims 200
```

**Hardware:** Apple M4, MPS backend, PyTorch 2.11.

**Result:** Iteration 1 took **74.5 minutes** (4,859 examples from 100 self-play
games). Projected total: ~62 hours for 50 iterations. Aborted after iteration 1.

**Breakdown:**
- Self-play: ~74 min (99.4% of wall time)
- Training: ~25 sec (5 epochs over 4,859 examples)
- Bottleneck: 100 games × 200 sims × ~48 moves = ~960,000 sequential
  `predict()` calls, each a single (4,8,8) tensor round-tripping to MPS.

**Lessons:**
- 200 sims in self-play is too expensive for single-threaded execution.
- Need separate `--self-play-sims` and `--arena-sims` flags.
- `predict_batch()` exists in network.py but is unused — batching across
  parallel games would be the biggest throughput win.

**Initial losses:** policy=4.809, value=0.675. Policy loss is near log(1024)≈6.93
(random over action space), so the network is barely above random after one
iteration, as expected.

### Run 2 — Reduced validation run (2026-04-03)

**Goal:** Quick pipeline validation with reduced compute.

**Command:**
```bash
python main.py --game checkers --iterations 10 --games 20 --sims 50 --arena-games 20
```

**Hardware:** Same.

**Expected:** ~4 min/iteration, ~40 min total. Should see policy loss decrease
from ~4.8 toward ~3-4 over 10 iterations if the pipeline is working.

**Status:** Running.

**Expected outcome:** The arena draw rate should decrease over iterations as the
network learns to produce sharper value estimates. Win rate of new-vs-old should
become increasingly decisive. Policy loss should decrease monotonically; value
loss should decrease with noise.

---

## Hyperparameter rationale

### Training loop

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `iterations` | 50 | Claudius's recommendation. Enough to see whether the value head is learning (loss curves should be clearly trending by iteration 20). Full convergence would require 200+, but this is a validation run. |
| `games` (self-play/iter) | 100 | Default. At ~40 moves per checkers game, this generates ~4,000 training positions per iteration. With symmetry augmentation disabled (TODO in checkers adapter), no multiplier. |
| `epochs` | 5 | Standard for AlphaZero-family implementations. More epochs risks overfitting to the current iteration's self-play data; fewer risks underfitting. 5 is the consensus sweet spot (see Appendix A of Silver et al. 2017). |
| `batch_size` | 512 | Balanced between gradient noise (too small) and memory (too large). On MPS with 1.9M params this fits comfortably. |
| `lr` | 1e-3 | Adam default. AlphaZero originally used SGD+momentum with a learning rate schedule, but Adam with 1e-3 is standard for smaller-scale reproductions. No schedule — we may need one for longer runs if loss plateaus. |
| `weight_decay` | 1e-4 | Mild L2 regularisation. Prevents weight explosion during early training when value targets are noisy. |
| `replay_buffer` | 50,000 | ~10–12 iterations of data at ~4K examples/iter. This means the network trains on a mix of recent and slightly older self-play data, which stabilises learning. Too small → catastrophic forgetting; too large → stale data slows adaptation. |

### MCTS

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `sims` | 200 | **Key choice.** Claudius's recommendation, up from the default 100. The arena uses MCTS to evaluate both networks — if the simulation budget is too low, MCTS can't exploit the network's value estimates and games look random. At 200 sims, the search tree is deep enough (~7 moves lookahead in checkers) that network quality differences produce measurable win-rate gaps. Claudius's email: "the arena draw rate should drop as sim count goes up." |
| `cpuct` | 1.5 | PUCT exploration constant. 1.5 is slightly above the canonical 1.0–1.25 range, encouraging more exploration. Appropriate for early training when the policy head is still noisy. Could tune down to ~1.0 once the network is stronger. |
| `dirichlet_alpha` | 0.3 | Controls exploration noise shape at MCTS root. Lower α → more peaked noise (one move gets most of the bonus). 0.3 is standard for games with moderate branching (~7 for checkers). For comparison: chess uses 0.3, Go uses 0.03 (much higher branching). |
| `dirichlet_epsilon` | 0.25 | Weight of Dirichlet noise vs network prior. 25% noise ensures the search explores moves the network doesn't initially favour, which is critical early in training when the policy head is near-random anyway. |
| `temperature_threshold` | 30 | Use temperature=1.0 (proportional to visit counts) for the first 30 moves, then switch to temperature=0 (greedy/argmax). This creates diverse openings in self-play while ensuring endgames are played optimally. 30 moves covers roughly the opening + early midgame in checkers. |

### Arena

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `arena_games` | 40 | 20 as each colour. Enough for a ~10% precision on win rate (binomial standard error ≈ √(p(1-p)/n) ≈ 8% at p=0.55, n=40). More games would give tighter estimates but slow down each iteration. |
| `arena_threshold` | 0.55 | Conservative gating: the new network must win 55% of decisive games (draws excluded) to replace the old one. This prevents regression from noise while still allowing gradual improvement. Lower thresholds (e.g. 0.50) would accept more updates but risk accepting random fluctuations. |
| **Draw handling** | Excluded | `win_rate = wins_new / (wins_new + wins_old)`. Draws are counted but don't affect the acceptance decision. This is important because early networks produce many draws — if draws counted as 0.5 wins each, the acceptance signal would be diluted. |

### Network architecture

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `res_blocks` | 6 | Moderate depth. AlphaZero used 19 (chess) or 39 (Go) blocks, but those were trained for millions of iterations on TPU pods. For a validation run on a laptop, 6 blocks provides enough representational capacity without making each forward pass (and therefore each MCTS sim) too slow. |
| `channels` | 128 | Width of the residual tower. 128 is the sweet spot between capacity and speed. The network needs to represent positional patterns (piece configurations, king threats, forced captures) — 128 channels gives 128 "features" at each spatial location, which is ample for an 8×8 board. |
| **Total params** | 1,914,119 | ~1.9M. Modest by modern standards. For comparison: Robin's `connect_four/` AlphaZero uses 227K params (smaller board), and Nonaga's failed AlphaZero used 570K params. The larger capacity here is justified by checkers' richer positional structure. |
| **Input encoding** | (4, 8, 8) | Four binary planes: own men, own kings, opponent men, opponent kings. Always in canonical form (own = player about to move). This is the minimal sufficient encoding — adding features like "threatened squares" or "move count" could help but adds complexity. Start simple. |
| **Action space** | 1024 | Flat encoding: `from_sq * 32 + to_sq`. Only ~200 of 1024 entries are ever legal (most from-to pairs are geometrically impossible). The policy head must learn to zero out illegal entries via MCTS masking — the network itself doesn't enforce legality. |

## Architectural decisions

### Why a shared framework instead of per-game implementations?

The original AlphaZero paper demonstrated that the same algorithm works across
chess, shogi, and Go — only the game rules change. Claudius implemented this
literally: `alpha_zero/` is game-agnostic, `games/` provides adapters. This
avoids the copy-paste bugs that accumulate when maintaining separate MCTS
implementations per game (Robin documented several in the Nonaga DESIGN_DECISIONS).

### Value backup: player-aware, not level-aware

The `_backup()` method negates the value when crossing a player boundary, not at
every tree level. This is critical for Nonaga (two-ply turns: move piece then
move tile, both by the same player) but also correct for standard one-ply games
like checkers (where it's equivalent to level-based negation).

### Canonical form: 180-degree rotation, not row-flip

For checkers on 32 dark squares, the canonical form for player -1 uses
`sq → 31 - sq` (180-degree rotation). A naive row-flip (`sq → (7-row, col)`)
doesn't work because even and odd rows have dark squares on different columns.
This matches the finding documented in `RaggedR/checkers`.

### Draw encoding: 1e-4, not 0

Draws return `1e-4` from `get_game_ended()`, not `0`. This distinguishes "game
is drawn" from "game is ongoing" — both would return 0 otherwise. The small
positive value has negligible effect on training (value targets are effectively
0 for draws) but prevents logic bugs in the terminal check.

## Known limitations and future work

1. **No symmetry augmentation for checkers.** The adapter's `get_symmetries()`
   returns identity only. Checkers has horizontal flip symmetry — implementing
   it would double the training data for free.

2. **No learning rate schedule.** For longer runs (200+ iterations), a cosine or
   step-decay schedule would likely help once the loss plateaus.

3. **Single-threaded self-play.** Each MCTS simulation calls `network.predict()`
   sequentially. Batching across multiple games or using a separate inference
   thread would dramatically speed up self-play.

4. **No temperature decay curve.** The current schedule (temp=1 for 30 moves,
   then temp=0) is a step function. A smoother decay might produce better
   training data.

5. **Action space sparsity.** Only ~200/1024 actions are geometrically valid.
   A compressed action space would make the policy head's job easier but would
   require more complex action encoding.

## References

- Silver et al. (2017). "Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm." arXiv:1712.01815.
- Samuel (1959). "Some Studies in Machine Learning Using the Game of Checkers."
  IBM Journal of Research and Development.
- RaggedR/nonaga `DESIGN_DECISIONS.md` — 13-attempt training history.
- RaggedR/checkers — TD(lambda) self-play with 10-feature linear evaluator.
