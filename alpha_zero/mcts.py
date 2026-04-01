"""
Monte Carlo Tree Search (AlphaZero variant).

AlphaZero MCTS uses a neural network instead of random rollouts:
  - Prior probabilities (P) from the policy head guide exploration.
  - Value estimates (V) from the value head replace rollout outcomes.
  - PUCT selection balances exploitation (Q) vs exploration (U = P / (1+N)).

Two-player zero-sum convention:
  Values are always from the perspective of the player who is about to move
  (i.e. the canonical player). When we back up a value through the tree we
  negate it each time we cross a player boundary (standard negamax).

Key fixes vs naive implementations:
  - Value sign is flipped on EVERY player change, not every tree level.
    (Important for two-ply games like Nonaga where multiple tree levels
    belong to the same player.)
  - Dirichlet noise is added only at the root, with configurable alpha and
    epsilon.
  - Terminal nodes short-circuit — no network call needed.

References:
  Silver et al. (2017) "Mastering Chess and Shogi by Self-Play with a
  General Reinforcement Learning Algorithm"
"""

import math
import numpy as np
from typing import Optional, Dict, Tuple, List

from alpha_zero.game import Game, State, Action


class MCTSNode:
    """A single node in the MCTS tree."""

    __slots__ = [
        "state", "player", "parent", "action_taken",
        "children", "visit_count", "value_sum",
        "prior", "_expanded",
    ]

    def __init__(
        self,
        state: State,
        player: int,
        parent: Optional["MCTSNode"] = None,
        action_taken: Optional[Action] = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.player = player          # player whose turn it is at this node
        self.parent = parent
        self.action_taken = action_taken
        self.children: List["MCTSNode"] = []
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self._expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return self._expanded

    def is_leaf(self) -> bool:
        return not self._expanded or len(self.children) == 0


class MCTS:
    """
    AlphaZero-style MCTS.

    Usage:
        mcts = MCTS(game, network, config)
        policy = mcts.get_action_probs(state, player, temperature=1.0)
        action = np.random.choice(len(policy), p=policy)
    """

    def __init__(self, game: Game, network, config):
        """
        Args:
            game:    Game instance (provides rules + state encoding).
            network: AlphaZeroNetwork instance (provides prior + value).
                     Can be None for uniform-prior random rollout (testing).
            config:  Config-like object with attributes:
                       num_mcts_sims  (int)   — simulations per move
                       cpuct          (float) — exploration constant
                       dirichlet_alpha (float) — Dirichlet noise alpha
                       dirichlet_epsilon (float) — noise weight at root
        """
        self.game = game
        self.network = network
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action_probs(
        self,
        state: State,
        player: int,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Run MCTS and return a policy vector over actions.

        Args:
            state:       Current game state.
            player:      Current player (+1 or -1).
            temperature: Controls sharpness of the returned policy.
                         1.0 = proportional to visit counts (exploration).
                         0.0 = argmax (greedy, used for evaluation).
            add_noise:   Add Dirichlet noise at root (True during self-play,
                         False during evaluation/arena).

        Returns:
            np.ndarray of shape (action_size,) summing to 1.
        """
        canonical = self.game.get_canonical_form(state, player)
        root = MCTSNode(canonical, player=1)  # canonical player is always +1
        self._expand(root)

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(self.config.num_mcts_sims):
            self._simulate(root)

        action_size = self.game.get_action_size()
        counts = np.zeros(action_size, dtype=np.float32)
        for child in root.children:
            idx = self.game.action_to_index(child.action_taken)
            counts[idx] = child.visit_count

        return self._counts_to_policy(counts, temperature)

    # ------------------------------------------------------------------
    # Core MCTS phases
    # ------------------------------------------------------------------

    def _simulate(self, root: MCTSNode):
        """Run one SELECT → EXPAND → EVALUATE → BACKUP simulation."""
        path = [root]
        node = root

        # SELECT: traverse tree using PUCT until we reach a leaf
        while node.is_expanded() and not self._is_terminal(node):
            node = self._select_child(node)
            path.append(node)

        # Check terminal
        terminal_value = self._get_terminal_value(node)
        if terminal_value is not None:
            self._backup(path, terminal_value)
            return

        # EXPAND + EVALUATE
        value = self._expand(node)

        # BACKUP
        self._backup(path, value)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """PUCT selection: argmax Q(s,a) + U(s,a)."""
        sqrt_total = math.sqrt(node.visit_count + 1)
        best_score = -float("inf")
        best_child = node.children[0]

        for child in node.children:
            q = child.q_value
            u = self.config.cpuct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> float:
        """
        Expand node: create children for all legal actions.

        Returns the value estimate for this position (from the neural net,
        or 0 if no network is provided).
        """
        if node._expanded:
            return node.q_value

        node._expanded = True

        valid_mask = self.game.get_valid_moves_mask(node.state, node.player)
        legal_actions = self.game.get_valid_moves(node.state, node.player)

        if not legal_actions:
            # No moves — treat as terminal (should be caught by get_game_ended)
            return 0.0

        # Get policy + value from network
        if self.network is not None:
            state_tensor = self.game.get_state_representation(node.state)
            policy, value = self.network.predict(state_tensor)
        else:
            # Uniform prior, zero value (for testing without a trained network)
            policy = np.ones(self.game.get_action_size(), dtype=np.float32)
            value = 0.0

        # Mask illegal actions and renormalise
        policy = policy * valid_mask.astype(np.float32)
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy /= policy_sum
        else:
            # All priors on illegal moves — fall back to uniform over legal
            policy[valid_mask] = 1.0 / valid_mask.sum()

        # Create child nodes
        for action in legal_actions:
            next_state, next_player = self.game.get_next_state(node.state, node.player, action)
            canonical_next = self.game.get_canonical_form(next_state, next_player)
            prior = float(policy[self.game.action_to_index(action)])
            child = MCTSNode(
                canonical_next,
                player=next_player,
                parent=node,
                action_taken=action,
                prior=prior,
            )
            node.children.append(child)

        return float(value)

    def _backup(self, path: List[MCTSNode], value: float):
        """
        Propagate value up the path, negating at each player boundary.

        value is from the perspective of the player at path[-1] (the leaf).
        """
        current_player = path[-1].player
        for node in reversed(path):
            if node.player != current_player:
                # Crossed a player boundary — flip sign
                value = -value
                current_player = node.player
            node.visit_count += 1
            node.value_sum += value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_terminal(self, node: MCTSNode) -> bool:
        result = self.game.get_game_ended(node.state, node.player)
        return result != 0

    def _get_terminal_value(self, node: MCTSNode) -> Optional[float]:
        result = self.game.get_game_ended(node.state, node.player)
        if result == 0:
            return None
        return float(result)

    def _add_dirichlet_noise(self, root: MCTSNode):
        """Add Dirichlet noise to root's children priors for exploration."""
        alpha = self.config.dirichlet_alpha
        epsilon = self.config.dirichlet_epsilon
        noise = np.random.dirichlet([alpha] * len(root.children))
        for child, n in zip(root.children, noise):
            child.prior = (1 - epsilon) * child.prior + epsilon * n

    @staticmethod
    def _counts_to_policy(counts: np.ndarray, temperature: float) -> np.ndarray:
        """Convert visit counts to a probability distribution."""
        if temperature == 0:
            policy = np.zeros_like(counts)
            policy[np.argmax(counts)] = 1.0
            return policy
        counts_temp = counts ** (1.0 / temperature)
        total = counts_temp.sum()
        if total == 0:
            # Fallback: uniform over all actions
            return np.ones_like(counts) / len(counts)
        return counts_temp / total
