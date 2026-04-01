"""
Abstract Game interface for the plug-and-play AlphaZero core.

Any game can be used with this AlphaZero implementation by subclassing Game
and implementing all abstract methods. The AlphaZero core (MCTS, network,
trainer) depends only on this interface — no game-specific logic leaks in.

Convention for player encoding:
  The canonical form always represents the board from the perspective of the
  player who is about to move. This makes the neural network's job simpler —
  it always predicts for "the current player" regardless of colour.

Convention for get_game_ended return value:
  0   — game not over
  1   — the player who just moved (or whose turn it is) has won
 -1   — the player who just moved has lost (opponent won)
  1e-4 — draw (small non-zero to distinguish from ongoing games in training)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Any, Tuple


State = Any   # game-specific state type
Action = Any  # game-specific action type


class Game(ABC):
    """
    Abstract base class defining the interface every game must implement.

    Implementations live in games/. The core AlphaZero machinery (MCTS,
    network, trainer) uses only the methods defined here.
    """

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_init_state(self) -> State:
        """Return the initial game state (before any moves are made)."""
        ...

    @abstractmethod
    def get_next_state(self, state: State, player: int, action: Action) -> Tuple[State, int]:
        """
        Apply action to state for player.

        Returns:
            (next_state, next_player)  —  next_player is +1 or -1.
        """
        ...

    @abstractmethod
    def get_valid_moves(self, state: State, player: int) -> List[Action]:
        """
        Return a list of legal actions for player in state.

        The list must be non-empty for non-terminal states. For terminal
        states the return value is undefined (callers check get_game_ended
        first).
        """
        ...

    @abstractmethod
    def get_game_ended(self, state: State, player: int) -> float:
        """
        Return the game outcome from player's perspective.

          0     — game not yet over
          1     — player wins
         -1     — player loses
          1e-4  — draw

        Called after every move; player is the player who just moved (or
        whose turn it is, depending on convention — be consistent).
        """
        ...

    @abstractmethod
    def get_canonical_form(self, state: State, player: int) -> State:
        """
        Return state from player's perspective.

        For two-player zero-sum games: flip the board representation so
        that "current player" is always player +1. The network always sees
        the board from the point of view of the player to move.

        For games that are already symmetric (e.g. player info encoded in
        state), this can be identity.
        """
        ...

    @abstractmethod
    def get_action_size(self) -> int:
        """
        Return the total number of possible actions (size of the action space).

        The policy head of the neural network outputs a vector of this size.
        Not all actions are legal in every state — get_valid_moves() is used
        to mask illegal ones.
        """
        ...

    @abstractmethod
    def get_state_representation(self, state: State) -> np.ndarray:
        """
        Encode state as a numpy array for the neural network.

        Typical shape: (C, H, W) for board-like games. The exact shape is
        implementation-defined; the network constructor reads it via
        game.get_state_representation(game.get_init_state()).shape.
        """
        ...

    @abstractmethod
    def get_display_name(self) -> str:
        """Return a human-readable name for the game (used in logs/reports)."""
        ...

    # ------------------------------------------------------------------
    # Optional helpers (have sensible defaults; override for efficiency)
    # ------------------------------------------------------------------

    def action_to_index(self, action: Action) -> int:
        """
        Convert a game-specific action to a flat integer index in [0, action_size).

        The default implementation assumes actions ARE integers. Override for
        structured action types (e.g. (from_sq, to_sq) tuples).
        """
        return int(action)

    def index_to_action(self, index: int) -> Action:
        """
        Inverse of action_to_index. Default: index is the action.
        """
        return index

    def get_valid_moves_mask(self, state: State, player: int) -> np.ndarray:
        """
        Return a boolean mask of shape (action_size,) where True means legal.

        Default: build mask from get_valid_moves(). Override for speed.
        """
        mask = np.zeros(self.get_action_size(), dtype=bool)
        for action in self.get_valid_moves(state, player):
            mask[self.action_to_index(action)] = True
        return mask

    def get_symmetries(self, state: State, pi: np.ndarray) -> List[Tuple[State, np.ndarray]]:
        """
        Return a list of (canonical_state, policy) pairs that are equivalent
        under the game's symmetry group (rotations, reflections, etc.).

        Used for data augmentation during training. Default: no symmetries
        (returns [(state, pi)]).

        pi: policy vector of shape (action_size,)
        """
        return [(state, pi)]

    def __repr__(self) -> str:
        return f"<Game: {self.get_display_name()}>"
