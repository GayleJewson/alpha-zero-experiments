"""
Nonaga game adapter for the plug-and-play AlphaZero core.

Nonaga is a two-player abstract strategy game played on a hexagonal grid.
Each player has 3 pieces. The goal is to arrange your 3 pieces into a
triangle (all mutually adjacent). Each turn has two plies:
  1. PIECE MOVE: slide one of your pieces along a hex direction.
  2. TILE MOVE: remove an unoccupied edge tile and place it elsewhere.

Source: RaggedR/nonaga — https://github.com/RaggedR/nonaga
Rules: https://www.steffen-spiele.de/fileadmin/media/Spiele/Nonaga/Nonaga_EN.pdf

Key AlphaZero challenge documented in the source repo:
  Standard single-model AlphaZero failed to produce a competent player.
  The root cause: Nonaga games cycle indefinitely under competent play,
  producing 80%+ draws in self-play. Draws give value=0, so the value head
  learns nothing, and the policy reinforces triangle-avoidance in a vicious
  cycle. Island-model MCTS with cross-play (ring topology, 30% cross-play)
  was the best neural approach, with 3/5 islands matching a 14-weight GA.

  This implementation is correct — the structural mismatch is a property of
  Nonaga itself, not an implementation bug. See DESIGN_DECISIONS.md in the
  source repo for the full 13-attempt saga.

Two-ply action encoding:
  Because each full turn is two sub-actions (piece move + tile move), this
  adapter flattens both into a single action space.

  Piece move: (from_tile, direction) — 49 tiles × 6 directions = 294 actions
  Tile move:  (src_tile, dst_tile)   — 49 × 49 = 2401 actions
  Total action space: 2695

  During a piece-move ply, only piece-move actions [0..293] are valid.
  During a tile-move ply, only tile-move actions [294..2694] are valid.
  This is exposed via get_valid_moves_mask().

State encoding (for the neural network):
  6 channels of shape 7×7 (the full hex grid bounding box):
    0: own pieces
    1: opponent pieces
    2: tile presence
    3: piece moved this turn (tile-move ply: marks origin of moved piece)
    4: last opponent moved tile (forbidden src this turn)
    5: ply type (all-zeros = piece move, all-ones = tile move)

  NOTE: The hex grid is stored in a 7×7 bounding box using axial coordinates.
  Not all 49 cells are valid hex positions.

Dependencies:
  This adapter can work standalone (no import of RaggedR/nonaga) but is
  designed to be drop-in compatible with the game engine in that repo.
  If you have a copy of RaggedR/nonaga, add game/ to sys.path and the
  _NonagaState class below can be replaced with the original NonagaState.
"""

import numpy as np
from copy import deepcopy
from enum import IntEnum
from typing import List, Tuple, Optional, FrozenSet, Dict

from alpha_zero.game import Game, State, Action


# ---------------------------------------------------------------------------
# Coordinate system
# ---------------------------------------------------------------------------

GRID_SIZE = 7  # 7×7 bounding box for the axial hex grid

# Valid axial (q, r) positions in a side-3 hex (19 cells for initial board)
# The hex is centered at (3, 3) with radius 2 (side 3).
def _build_valid_set():
    valid = set()
    for r in range(GRID_SIZE):
        for q in range(GRID_SIZE):
            if max(abs(q - 3), abs(r - 3), abs((q + r) - 6)) <= 3:
                valid.add((q, r))
    return frozenset(valid)

VALID_QR: FrozenSet[Tuple[int, int]] = _build_valid_set()

def qr_to_idx(q: int, r: int) -> int:
    return r * GRID_SIZE + q

def idx_to_qr(idx: int) -> Tuple[int, int]:
    return idx % GRID_SIZE, idx // GRID_SIZE

# Axial hex directions
HEX_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
NUM_DIRS = len(HEX_DIRS)

# Precompute adjacency: idx -> list of neighbor idxs
NEIGHBORS: Dict[int, List[int]] = {}
for q, r in VALID_QR:
    idx = qr_to_idx(q, r)
    nbrs = []
    for dq, dr in HEX_DIRS:
        nq, nr = q + dq, r + dr
        if (nq, nr) in VALID_QR:
            nbrs.append(qr_to_idx(nq, nr))
    NEIGHBORS[idx] = nbrs

# Index all valid positions
VALID_INDICES = sorted(NEIGHBORS.keys())
VALID_INDEX_SET = set(VALID_INDICES)

# Initial 19-tile hex (side 2 from center — the starting board is smaller)
INITIAL_TILES: FrozenSet[int] = frozenset(
    qr_to_idx(q, r)
    for (q, r) in VALID_QR
    if max(abs(q - 3), abs(r - 3), abs((q + r) - 6)) <= 2
)

# Initial piece positions: alternating corners of the 19-hex
_CORNERS = [(5, 1), (5, 3), (3, 5), (1, 5), (1, 3), (3, 1)]
INITIAL_PIECES = {
    1:  frozenset(qr_to_idx(*_CORNERS[i]) for i in [0, 2, 4]),
    -1: frozenset(qr_to_idx(*_CORNERS[i]) for i in [1, 3, 5]),
}

# Action space sizes
PIECE_ACTION_SIZE = len(VALID_INDICES) * NUM_DIRS  # 49 * 6 = 294 (but most are illegal)
TILE_ACTION_SIZE  = len(VALID_INDICES) * len(VALID_INDICES)  # 49 * 49 = 2401
TOTAL_ACTION_SIZE = PIECE_ACTION_SIZE + TILE_ACTION_SIZE

PIECE_PLY = 0
TILE_PLY  = 1


# ---------------------------------------------------------------------------
# Game state (standalone, not importing from RaggedR/nonaga)
# ---------------------------------------------------------------------------

class NonagaState:
    """Lightweight game state. Compatible with RaggedR/nonaga's NonagaState API."""

    __slots__ = [
        "tiles", "pieces", "current_player", "ply_type",
        "last_moved_tile", "piece_moved_from", "winner",
    ]

    def __init__(self):
        self.tiles: FrozenSet[int] = INITIAL_TILES
        self.pieces: Dict[int, FrozenSet[int]] = {
            1: INITIAL_PIECES[1],
            -1: INITIAL_PIECES[-1],
        }
        self.current_player: int = 1
        self.ply_type: int = PIECE_PLY
        self.last_moved_tile: Optional[int] = None
        self.piece_moved_from: Optional[int] = None
        self.winner: Optional[int] = None

    def copy(self) -> "NonagaState":
        s = NonagaState.__new__(NonagaState)
        s.tiles = self.tiles
        s.pieces = dict(self.pieces)
        s.current_player = self.current_player
        s.ply_type = self.ply_type
        s.last_moved_tile = self.last_moved_tile
        s.piece_moved_from = self.piece_moved_from
        s.winner = self.winner
        return s

    @property
    def occupied(self) -> FrozenSet[int]:
        return self.pieces[1] | self.pieces[-1]

    def is_terminal(self) -> bool:
        return self.winner is not None

    def _check_triangle(self) -> Optional[int]:
        """Return the winning player if either has 3 mutually adjacent pieces."""
        for player in (1, -1):
            pcs = list(self.pieces[player])
            if len(pcs) < 3:
                continue
            a, b, c = pcs
            na = set(NEIGHBORS.get(a, []))
            nb = set(NEIGHBORS.get(b, []))
            if b in na and c in na and c in nb:
                return player
        return None

    def get_piece_moves(self) -> List[int]:
        """
        Return list of piece-move action indices (flat) legal for current player.
        A piece slides to an adjacent tile if that tile is occupied and it
        can continue sliding until it reaches an empty tile.

        TODO: Implement full slide mechanics from RaggedR/nonaga game/nonaga.py.
              For now returns adjacent-step moves as a stub.
        """
        # TODO: replace with slide() logic from RaggedR/nonaga
        moves = []
        player = self.current_player
        for piece in self.pieces[player]:
            q, r = idx_to_qr(piece)
            for dir_idx, (dq, dr) in enumerate(HEX_DIRS):
                nq, nr = q + dq, r + dr
                nidx = qr_to_idx(nq, nr)
                if (nq, nr) in VALID_QR and nidx in self.tiles and nidx not in self.occupied:
                    # Encode as piece_idx * NUM_DIRS + dir_idx
                    piece_pos = VALID_INDICES.index(piece) if piece in VALID_INDEX_SET else -1
                    if piece_pos >= 0:
                        moves.append(piece_pos * NUM_DIRS + dir_idx)
        return moves

    def get_tile_moves(self) -> List[int]:
        """
        Return list of tile-move action indices (flat, offset by PIECE_ACTION_SIZE).

        An edge tile can be removed if:
          - It has no piece on it
          - Its removal keeps the remaining tiles connected
          - It is not the tile the opponent just moved (last_moved_tile)

        It can be placed at any valid position adjacent to the remaining tiles
        that is not currently a tile.

        TODO: Import full connectivity check from RaggedR/nonaga.
        """
        # TODO: use _removable_edge_tiles() and _edge_positions() from NonagaState
        moves = []
        edge_tiles = self._edge_tiles()
        edge_positions = self._edge_positions()
        for src in edge_tiles:
            if src in self.occupied:
                continue
            if src == self.last_moved_tile:
                continue
            # Connectivity check stub — assume all edge tiles are removable
            # TODO: call self._is_connected_without(src)
            src_pos = VALID_INDICES.index(src) if src in VALID_INDEX_SET else -1
            if src_pos < 0:
                continue
            for dst in edge_positions:
                if dst == src:
                    continue
                dst_pos = VALID_INDICES.index(dst) if dst in VALID_INDEX_SET else -1
                if dst_pos < 0:
                    continue
                flat = src_pos * len(VALID_INDICES) + dst_pos
                moves.append(PIECE_ACTION_SIZE + flat)
        return moves

    def _edge_tiles(self) -> List[int]:
        edge = []
        for idx in self.tiles:
            for nbr in NEIGHBORS.get(idx, []):
                if nbr not in self.tiles:
                    edge.append(idx)
                    break
        return edge

    def _edge_positions(self) -> List[int]:
        positions = set()
        for idx in self.tiles:
            for nbr in NEIGHBORS.get(idx, []):
                if nbr not in self.tiles and nbr in VALID_INDEX_SET:
                    positions.add(nbr)
        return list(positions)

    def apply_piece_move(self, action_idx: int) -> "NonagaState":
        """Apply a piece-move action and return new state (tile-move ply)."""
        piece_pos = action_idx // NUM_DIRS
        dir_idx   = action_idx % NUM_DIRS
        piece_idx = VALID_INDICES[piece_pos]
        q, r = idx_to_qr(piece_idx)
        dq, dr = HEX_DIRS[dir_idx]
        nq, nr = q + dq, r + dr
        new_piece = qr_to_idx(nq, nr)

        s = self.copy()
        old_set = s.pieces[s.current_player]
        s.pieces[s.current_player] = (old_set - {piece_idx}) | {new_piece}
        s.piece_moved_from = piece_idx
        s.ply_type = TILE_PLY

        # Check for win after piece move
        winner = s._check_triangle()
        if winner is not None:
            s.winner = winner
        return s

    def apply_tile_move(self, action_idx: int) -> "NonagaState":
        """Apply a tile-move action and return new state (next player's piece-move ply)."""
        flat = action_idx - PIECE_ACTION_SIZE
        src_pos = flat // len(VALID_INDICES)
        dst_pos = flat % len(VALID_INDICES)
        src_idx = VALID_INDICES[src_pos]
        dst_idx = VALID_INDICES[dst_pos]

        s = self.copy()
        s.tiles = (s.tiles - {src_idx}) | {dst_idx}
        s.last_moved_tile = dst_idx
        s.piece_moved_from = None
        s.current_player = -s.current_player
        s.ply_type = PIECE_PLY
        return s


# ---------------------------------------------------------------------------
# Game adapter
# ---------------------------------------------------------------------------

class NonagaGame(Game):
    """
    Nonaga adapter for the AlphaZero plug-and-play interface.

    Informed by RaggedR/nonaga (https://github.com/RaggedR/nonaga).
    The two-ply structure (piece move + tile move per turn) is handled by
    encoding both sub-actions in a single flattened action space.
    """

    def get_display_name(self) -> str:
        return "Nonaga"

    def get_action_size(self) -> int:
        return TOTAL_ACTION_SIZE  # 2695

    def get_init_state(self) -> NonagaState:
        return NonagaState()

    def get_next_state(
        self, state: NonagaState, player: int, action: int
    ) -> Tuple[NonagaState, int]:
        if state.ply_type == PIECE_PLY:
            next_state = state.apply_piece_move(action)
            # Same player continues to tile-move ply
            return next_state, state.current_player
        else:
            next_state = state.apply_tile_move(action)
            return next_state, next_state.current_player

    def get_valid_moves(self, state: NonagaState, player: int) -> List[int]:
        if state.is_terminal():
            return []
        if state.ply_type == PIECE_PLY:
            return state.get_piece_moves()
        else:
            return state.get_tile_moves()

    def get_game_ended(self, state: NonagaState, player: int) -> float:
        if state.winner is None:
            return 0
        if state.winner == player:
            return 1.0
        return -1.0

    def get_canonical_form(self, state: NonagaState, player: int) -> NonagaState:
        """
        Return state from player's perspective.
        If player == -1, swap pieces and invert current_player.
        """
        if player == 1:
            return state
        s = state.copy()
        s.pieces = {1: state.pieces[-1], -1: state.pieces[1]}
        s.current_player = -state.current_player
        if state.winner is not None:
            s.winner = -state.winner
        return s

    def get_state_representation(self, state: NonagaState) -> np.ndarray:
        """
        Encode Nonaga state as 6-channel 7×7 tensor.

        Channels:
          0: own pieces (current_player == 1 in canonical form)
          1: opponent pieces
          2: tile presence
          3: piece_moved_from this turn (marks ply transition origin)
          4: last_moved_tile (forbidden as tile-move source)
          5: ply type (0 = piece move ply, 1 = tile move ply)
        """
        planes = np.zeros((6, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        for idx in state.pieces.get(1, []):
            q, r = idx_to_qr(idx)
            planes[0, r, q] = 1.0

        for idx in state.pieces.get(-1, []):
            q, r = idx_to_qr(idx)
            planes[1, r, q] = 1.0

        for idx in state.tiles:
            q, r = idx_to_qr(idx)
            planes[2, r, q] = 1.0

        if state.piece_moved_from is not None:
            q, r = idx_to_qr(state.piece_moved_from)
            planes[3, r, q] = 1.0

        if state.last_moved_tile is not None:
            q, r = idx_to_qr(state.last_moved_tile)
            planes[4, r, q] = 1.0

        if state.ply_type == TILE_PLY:
            planes[5, :, :] = 1.0

        return planes

    def action_to_index(self, action: int) -> int:
        return action

    def index_to_action(self, index: int) -> int:
        return index
