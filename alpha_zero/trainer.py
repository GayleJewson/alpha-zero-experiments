"""
AlphaZero training loop.

Three phases per iteration:
  1. Self-play      — generate game data using MCTS + current network.
  2. Train          — update network on the generated data.
  3. Arena          — pit new network against old; keep new if win rate > threshold.

The loop is game-agnostic: it uses only the Game interface and the network's
predict() method. All game-specific details are hidden behind those APIs.

Data format:
  Each position generates a training example:
    (state_repr, policy_target, value_target)
  where:
    state_repr:    np.ndarray (C, H, W)
    policy_target: np.ndarray (action_size,) — MCTS visit-count distribution
    value_target:  float in {-1, 0, 1} — game outcome from this player's view

Usage:
    trainer = Trainer(game, config)
    trainer.learn()
"""

import os
import time
import math
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from typing import List, Tuple, Deque, Optional
from dataclasses import dataclass, field

from alpha_zero.game import Game, State
from alpha_zero.mcts import MCTS
from alpha_zero.network import AlphaZeroNetwork

logger = logging.getLogger(__name__)

Example = Tuple[np.ndarray, np.ndarray, float]


@dataclass
class TrainerConfig:
    """All hyperparameters for the training loop."""

    # Self-play
    num_iterations: int = 100
    num_self_play_games: int = 100
    replay_buffer_size: int = 50_000   # max examples to keep across iterations

    # MCTS
    num_mcts_sims: int = 100
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3        # lower for games with many legal moves
    dirichlet_epsilon: float = 0.25

    # Training
    batch_size: int = 512
    num_epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0      # relative weight of value loss vs policy loss

    # Arena
    arena_games: int = 40
    arena_win_threshold: float = 0.55   # fraction of wins needed to replace old net

    # Temperature schedule
    temperature_threshold: int = 30     # use temperature=1 for first N moves, then 0

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10          # save every N iterations

    # Network
    num_res_blocks: int = 6
    num_channels: int = 128


class SelfPlayGame:
    """Plays one game of self-play and collects (state, pi, v) examples."""

    def __init__(self, game: Game, mcts: MCTS, config: TrainerConfig):
        self.game = game
        self.mcts = mcts
        self.config = config

    def play(self) -> List[Example]:
        """
        Play a full game via MCTS self-play.

        Returns a list of (state_repr, policy, outcome) examples.
        The outcome is filled in at the end of the game once we know who won.
        """
        examples_buffer = []  # (state_repr, policy, current_player)
        state = self.game.get_init_state()
        player = 1
        move_count = 0

        while True:
            canonical = self.game.get_canonical_form(state, player)
            temperature = 1.0 if move_count < self.config.temperature_threshold else 0.0

            policy = self.mcts.get_action_probs(state, player, temperature=temperature, add_noise=True)

            # Data augmentation via symmetries
            symmetries = self.game.get_symmetries(canonical, policy)
            for sym_state, sym_policy in symmetries:
                state_repr = self.game.get_state_representation(sym_state)
                examples_buffer.append((state_repr, sym_policy, player))

            # Sample action from policy
            action_idx = np.random.choice(len(policy), p=policy)
            action = self.game.index_to_action(action_idx)

            state, player = self.game.get_next_state(state, player, action)
            move_count += 1

            result = self.game.get_game_ended(state, player)
            if result != 0:
                # Assign outcomes. result is from `player`'s perspective.
                # For each buffered example, the outcome is +result if the example
                # was recorded when that example's player matches the winner, else -result.
                winner_player = player if result > 0 else -player
                examples: List[Example] = []
                for (s_repr, pi, p) in examples_buffer:
                    outcome = result if p == player else -result
                    # Clamp to {-1, 0, 1} — draw uses 1e-4 but we keep it as is
                    examples.append((s_repr, pi, float(outcome)))
                return examples


class Arena:
    """Pits two networks against each other to determine which is stronger."""

    def __init__(self, game: Game, config: TrainerConfig):
        self.game = game
        self.config = config

    def play_games(self, net_new, net_old, num_games: int) -> Tuple[int, int, int]:
        """
        Play num_games between net_new and net_old.

        Returns (wins_new, wins_old, draws).
        Half the games net_new plays as player +1, half as player -1.
        """
        wins_new = wins_old = draws = 0

        for game_idx in range(num_games):
            # Alternate who goes first
            if game_idx % 2 == 0:
                new_player, old_player = 1, -1
            else:
                new_player, old_player = -1, 1

            mcts_new = MCTS(self.game, net_new, self.config)
            mcts_old = MCTS(self.game, net_old, self.config)

            state = self.game.get_init_state()
            player = 1

            while True:
                if player == new_player:
                    pi = mcts_new.get_action_probs(state, player, temperature=0, add_noise=False)
                else:
                    pi = mcts_old.get_action_probs(state, player, temperature=0, add_noise=False)

                action_idx = np.argmax(pi)
                action = self.game.index_to_action(action_idx)
                state, player = self.game.get_next_state(state, player, action)
                result = self.game.get_game_ended(state, player)

                if result != 0:
                    if abs(result) < 0.1:  # draw
                        draws += 1
                    elif result > 0:
                        # `player` won
                        if player == new_player:
                            wins_new += 1
                        else:
                            wins_old += 1
                    else:
                        if player == new_player:
                            wins_old += 1
                        else:
                            wins_new += 1
                    break

        return wins_new, wins_old, draws


class Trainer:
    """
    Manages the full AlphaZero training loop:
      self-play → train network → arena → iterate.
    """

    def __init__(self, game: Game, config: Optional[TrainerConfig] = None):
        self.game = game
        self.config = config or TrainerConfig()
        self.device = self._get_device()

        # Build network
        init_state = game.get_init_state()
        input_shape = game.get_state_representation(
            game.get_canonical_form(init_state, 1)
        ).shape
        action_size = game.get_action_size()

        self.network = AlphaZeroNetwork(
            input_shape=input_shape,
            action_size=action_size,
            num_res_blocks=self.config.num_res_blocks,
            num_channels=self.config.num_channels,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.replay_buffer: Deque[Example] = deque(maxlen=self.config.replay_buffer_size)

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        logger.info(
            f"Trainer ready | game={game.get_display_name()} | "
            f"input={input_shape} | actions={action_size} | "
            f"params={self.network.count_parameters():,} | device={self.device}"
        )

    def learn(self):
        """Run the full training loop."""
        for iteration in range(1, self.config.num_iterations + 1):
            logger.info(f"=== Iteration {iteration}/{self.config.num_iterations} ===")
            t0 = time.time()

            # 1. Self-play
            new_examples = self._self_play()
            self.replay_buffer.extend(new_examples)
            logger.info(f"  Self-play: {len(new_examples)} examples | buffer={len(self.replay_buffer)}")

            # 2. Train
            self._train()

            # 3. Arena (skip first iteration — no old weights to compare)
            if iteration > 1:
                accepted = self._arena()
                logger.info(f"  Arena: {'accepted new' if accepted else 'kept old'} network")

            elapsed = time.time() - t0
            logger.info(f"  Iteration time: {elapsed:.1f}s")

            if iteration % self.config.checkpoint_every == 0:
                self._save_checkpoint(iteration)

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------

    def _self_play(self) -> List[Example]:
        mcts = MCTS(self.game, self.network, self.config)
        sp = SelfPlayGame(self.game, mcts, self.config)
        examples = []
        for _ in range(self.config.num_self_play_games):
            examples.extend(sp.play())
        return examples

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self):
        self.network.train()
        data = list(self.replay_buffer)
        random.shuffle(data)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.num_epochs):
            for i in range(0, len(data), self.config.batch_size):
                batch = data[i : i + self.config.batch_size]
                if not batch:
                    continue

                states = np.array([e[0] for e in batch], dtype=np.float32)
                target_pis = np.array([e[1] for e in batch], dtype=np.float32)
                target_vs = np.array([e[2] for e in batch], dtype=np.float32)

                states_t    = torch.tensor(states).to(self.device)
                target_pis_t = torch.tensor(target_pis).to(self.device)
                target_vs_t  = torch.tensor(target_vs).to(self.device)

                policy_logits, values = self.network(states_t)

                # Policy loss: cross-entropy between MCTS distribution and network output
                log_probs = F.log_softmax(policy_logits, dim=1)
                policy_loss = -(target_pis_t * log_probs).sum(dim=1).mean()

                # Value loss: MSE between network prediction and game outcome
                value_loss = F.mse_loss(values.squeeze(1), target_vs_t)

                loss = policy_loss + self.config.value_loss_weight * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        if num_batches > 0:
            logger.info(
                f"  Train: policy_loss={total_policy_loss/num_batches:.4f} "
                f"value_loss={total_value_loss/num_batches:.4f}"
            )

    # ------------------------------------------------------------------
    # Arena
    # ------------------------------------------------------------------

    def _arena(self) -> bool:
        """
        Compare new network against a snapshot of itself from before training.
        Returns True if the new network is accepted.
        """
        # Load the previous checkpoint if it exists
        old_net_path = os.path.join(self.config.checkpoint_dir, "best.pt")
        if not os.path.exists(old_net_path):
            # No previous checkpoint — auto-accept
            self._save_best()
            return True

        init_state = self.game.get_init_state()
        input_shape = self.game.get_state_representation(
            self.game.get_canonical_form(init_state, 1)
        ).shape
        old_net = AlphaZeroNetwork(
            input_shape=input_shape,
            action_size=self.game.get_action_size(),
            num_res_blocks=self.config.num_res_blocks,
            num_channels=self.config.num_channels,
        ).to(self.device)
        old_net.load_state_dict(torch.load(old_net_path, map_location=self.device))

        arena = Arena(self.game, self.config)
        wins_new, wins_old, draws = arena.play_games(
            self.network, old_net, self.config.arena_games
        )
        logger.info(f"  Arena results: new={wins_new} old={wins_old} draws={draws}")

        decisive = wins_new + wins_old
        if decisive == 0:
            win_rate = 0.5  # all draws
        else:
            win_rate = wins_new / decisive

        accepted = win_rate >= self.config.arena_win_threshold
        if accepted:
            self._save_best()
        return accepted

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, iteration: int):
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{iteration:04d}.pt")
        torch.save(self.network.state_dict(), path)
        logger.info(f"  Saved checkpoint: {path}")

    def _save_best(self):
        path = os.path.join(self.config.checkpoint_dir, "best.pt")
        torch.save(self.network.state_dict(), path)

    def load_checkpoint(self, path: str):
        self.network.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        logger.info(f"Loaded checkpoint: {path}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
