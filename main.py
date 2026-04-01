"""
Entry point for AlphaZero training experiments.

Usage:
    python main.py --game nonaga
    python main.py --game checkers
    python main.py --game blokus
    python main.py --game checkers --iterations 50 --sims 200 --channels 128
    python main.py --game checkers --load checkpoints/best.pt --eval-only

All hyperparameters have sensible defaults. Pass --help for the full list.
"""

import argparse
import logging
import sys

from alpha_zero.utils import setup_logging
from alpha_zero.trainer import Trainer, TrainerConfig


def build_game(name: str):
    name = name.lower().strip()
    if name in ("nonaga", "noriega"):
        from games.nonaga import NonagaGame
        return NonagaGame()
    elif name == "checkers":
        from games.checkers import CheckersGame
        return CheckersGame()
    elif name == "blokus":
        from games.blokus import BlokusGame
        return BlokusGame()
    else:
        print(f"Unknown game: {name!r}. Choose from: nonaga, checkers, blokus", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Plug-and-play AlphaZero: train on any supported game.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--game", default="checkers",
                        choices=["nonaga", "noriega", "checkers", "blokus"],
                        help="Game to train on.")

    # Training loop
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations (self-play + train + arena).")
    parser.add_argument("--games", type=int, default=100,
                        help="Self-play games per iteration.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs per iteration.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--replay", type=int, default=50_000,
                        help="Replay buffer size (examples).")

    # MCTS
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS simulations per move.")
    parser.add_argument("--cpuct", type=float, default=1.5,
                        help="PUCT exploration constant.")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet noise alpha (lower = more peaked exploration).")

    # Arena
    parser.add_argument("--arena-games", type=int, default=40,
                        help="Number of arena games per iteration.")
    parser.add_argument("--arena-threshold", type=float, default=0.55,
                        help="Win fraction required for new network to replace old.")

    # Network
    parser.add_argument("--res-blocks", type=int, default=6,
                        help="Number of residual blocks in the network.")
    parser.add_argument("--channels", type=int, default=128,
                        help="Hidden channels in the residual tower.")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory for saving checkpoints.")
    parser.add_argument("--load", default=None,
                        help="Path to a checkpoint to load before training.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; just run the arena against a random player.")

    # Logging
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")

    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    game = build_game(args.game)
    logger.info(f"Game: {game.get_display_name()}")
    logger.info(f"Action space: {game.get_action_size()}")

    config = TrainerConfig(
        num_iterations=args.iterations,
        num_self_play_games=args.games,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        replay_buffer_size=args.replay,
        num_mcts_sims=args.sims,
        cpuct=args.cpuct,
        dirichlet_alpha=args.dirichlet_alpha,
        arena_games=args.arena_games,
        arena_win_threshold=args.arena_threshold,
        num_res_blocks=args.res_blocks,
        num_channels=args.channels,
        checkpoint_dir=f"{args.checkpoint_dir}/{args.game}",
    )

    trainer = Trainer(game, config)

    if args.load:
        trainer.load_checkpoint(args.load)

    if args.eval_only:
        logger.info("Eval-only mode: running arena vs random player (no network).")
        from alpha_zero.mcts import MCTS
        from alpha_zero.trainer import Arena
        import numpy as np

        # A "random" network: uniform policy, zero value
        class RandomNet:
            def predict(self, state_array):
                action_size = game.get_action_size()
                return np.ones(action_size) / action_size, 0.0

        random_net = RandomNet()
        arena = Arena(game, config)
        wins, losses, draws = arena.play_games(trainer.network, random_net, config.arena_games)
        logger.info(f"vs random: wins={wins} losses={losses} draws={draws}")
    else:
        trainer.learn()


if __name__ == "__main__":
    main()
