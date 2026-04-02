"""
Checkers (Draughts) game adapter for the plug-and-play AlphaZero core.

Standard 8×8 English draughts on 32 dark squares.
  - Black moves first (player +1), White is player -1.
  - Black moves downward (toward row 7), White moves upward (toward row 0).
  - Mandatory captures (if you can jump, you must).
  - Multi-jump chains via DFS.
  - King promotion on back row.
  - Draw after 80 half-moves without a capture.

Source: RaggedR/checkers — https://github.com/RaggedR/checkers
AI approach in source: TD(lambda) self-play with a 10-feature linear evaluator
  (not AlphaZero). This adapter wraps the same game rules for AlphaZero training.

Board representation:
  32 playable (dark) squares, indexed 0–31 (top-left to bottom-right).
  Piece encoding: 0=empty, 1=black man, 2=black king, -1=white man, -2=white king.

Action encoding:
  An action is a move sequence: [from_sq, sq1, sq2, ...] for multi-jumps.
  We flatten into an integer: from_sq * 32 + to_sq (first + last square only).
  This gives a 32×32=1024-dimensional action space.

  Multi-jump disambiguation: if multiple jump chains share the same
  (from, final) pair, we take the first found (DFS order). This is a known
  approximation — full disambiguation would require encoding the full path,
  which is rare in practice.

State encoding for neural network (8×8, 4 channels):
  0: black men
  1: black kings
  2: white men (negated for canonical — own pieces)
  3: white kings (negated for canonical — own kings)

  In canonical form (always from current player's perspective), planes 0-1
  are own pieces and planes 2-3 are opponent pieces, with the board
  optionally flipped.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from copy import deepcopy

from alpha_zero.game import Game, State, Action


# ---------------------------------------------------------------------------
# Board geometry (precomputed at import time, matching RaggedR/checkers)
# ---------------------------------------------------------------------------

SQ_TO_RC: dict = {}
RC_TO_SQ: dict = {}
for _sq in range(32):
    _r = _sq // 4
    _c = 2 * (_sq % 4) + (1 if _r % 2 == 0 else 0)
    SQ_TO_RC[_sq] = (_r, _c)
    RC_TO_SQ[(_r, _c)] = _sq

_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

ADJACENT: List[List[Tuple]] = [[] for _ in range(32)]   # sq -> [(dir, nbr_sq)]
JUMPS: List[List[Tuple]] = [[] for _ in range(32)]       # sq -> [(dir, over_sq, land_sq)]

for _sq in range(32):
    _r, _c = SQ_TO_RC[_sq]
    for _dr, _dc in _DIRS:
        _nr, _nc = _r + _dr, _c + _dc
        if 0 <= _nr < 8 and 0 <= _nc < 8:
            _nsq = RC_TO_SQ.get((_nr, _nc))
            if _nsq is not None:
                ADJACENT[_sq].append(((_dr, _dc), _nsq))
        _jr, _jc = _r + 2 * _dr, _c + 2 * _dc
        if 0 <= _jr < 8 and 0 <= _jc < 8:
            _over = RC_TO_SQ.get((_nr, _nc))
            _land = RC_TO_SQ.get((_jr, _jc))
            if _over is not None and _land is not None:
                JUMPS[_sq].append(((_dr, _dc), _over, _land))

BLACK_KING_ROW = 7
WHITE_KING_ROW = 0
DRAW_HALF_MOVES = 80

ACTION_SIZE = 32 * 32  # flat (from, to) index


# ---------------------------------------------------------------------------
# Game state (mirrors RaggedR/checkers/checkers.py)
# ---------------------------------------------------------------------------

class CheckersState:
    """
    Checkers game state.

    Compatible with the CheckersGame API in RaggedR/checkers/checkers.py,
    adapted slightly for the AlphaZero interface.
    """

    def __init__(self):
        self.board = [0] * 32
        for sq in range(12):
            self.board[sq] = 1    # black man
        for sq in range(20, 32):
            self.board[sq] = -1   # white man
        self.turn = 1             # +1 = black, -1 = white
        self.half_moves_since_capture = 0

    def copy(self) -> "CheckersState":
        s = CheckersState.__new__(CheckersState)
        s.board = self.board[:]
        s.turn = self.turn
        s.half_moves_since_capture = self.half_moves_since_capture
        return s

    def _forward_dirs(self, color: int) -> List[int]:
        return [1] if color == 1 else [-1]

    def get_legal_moves(self) -> List[List[int]]:
        """
        Return all legal move sequences for the current player.
        Mandatory captures: if any jump exists, only jumps are returned.
        """
        jumps = self._get_jump_sequences()
        if jumps:
            return jumps
        return self._get_simple_moves()

    def _get_simple_moves(self) -> List[List[int]]:
        moves = []
        for sq in range(32):
            piece = self.board[sq]
            if piece == 0 or (piece > 0) != (self.turn > 0):
                continue
            is_king = abs(piece) == 2
            allowed_dr = self._forward_dirs(self.turn) if not is_king else [-1, 1]
            for (dr, dc), nsq in ADJACENT[sq]:
                if dr not in allowed_dr:
                    continue
                if self.board[nsq] == 0:
                    moves.append([sq, nsq])
        return moves

    def _get_jump_sequences(self) -> List[List[int]]:
        all_jumps: List[List[int]] = []
        for sq in range(32):
            piece = self.board[sq]
            if piece == 0 or (piece > 0) != (self.turn > 0):
                continue
            self._dfs_jumps(sq, piece, self.board[:], [sq], set(), all_jumps)
        return all_jumps

    def _dfs_jumps(
        self,
        sq: int,
        piece: int,
        board: List[int],
        path: List[int],
        captured: Set[int],
        result: List[List[int]],
    ):
        is_king = abs(piece) == 2
        allowed_dr = self._forward_dirs(self.turn) if not is_king else [-1, 1]
        found = False
        for (dr, dc), over, land in JUMPS[sq]:
            if dr not in allowed_dr:
                continue
            if over in captured:
                continue
            over_piece = board[over]
            if over_piece == 0 or (over_piece > 0) == (self.turn > 0):
                continue
            if board[land] != 0 and land != path[0]:  # land free (or back to start for loop)
                continue
            if board[land] != 0:
                continue
            found = True
            new_board = board[:]
            new_board[over] = 0
            new_board[land] = piece
            new_board[sq] = 0 if land != sq else piece
            self._dfs_jumps(land, piece, new_board, path + [land], captured | {over}, result)
        if not found:
            if len(path) > 1:
                result.append(path[:])

    def apply_move(self, move: List[int]) -> "CheckersState":
        """Apply a move sequence and return the new state."""
        s = self.copy()
        from_sq = move[0]
        to_sq = move[-1]
        piece = s.board[from_sq]

        is_capture = len(move) > 2 or (
            len(move) == 2 and abs(SQ_TO_RC[from_sq][0] - SQ_TO_RC[to_sq][0]) == 2
        )

        # For jump sequences, clear captured pieces
        if is_capture:
            s.half_moves_since_capture = 0
            for i in range(len(move) - 1):
                r0, c0 = SQ_TO_RC[move[i]]
                r1, c1 = SQ_TO_RC[move[i + 1]]
                over_r = (r0 + r1) // 2
                over_c = (c0 + c1) // 2
                over_sq = RC_TO_SQ.get((over_r, over_c))
                if over_sq is not None:
                    s.board[over_sq] = 0
        else:
            s.half_moves_since_capture += 1

        s.board[from_sq] = 0
        s.board[to_sq] = piece

        # King promotion
        r_to, _ = SQ_TO_RC[to_sq]
        if piece == 1 and r_to == BLACK_KING_ROW:
            s.board[to_sq] = 2
        elif piece == -1 and r_to == WHITE_KING_ROW:
            s.board[to_sq] = -2

        s.turn = -s.turn
        return s

    def is_terminal(self) -> Tuple[bool, float]:
        """
        Returns (is_terminal, result_for_previous_player).
        result: 1 = previous player won, 0 = draw, -1 = loss.
        """
        if self.half_moves_since_capture >= DRAW_HALF_MOVES:
            return True, 1e-4  # draw

        moves = self.get_legal_moves()
        if not moves:
            # Current player has no moves — they lose
            return True, 1.0  # from the *previous* player's perspective: win

        return False, 0.0

    def to_8x8(self) -> np.ndarray:
        """Return an 8×8 board with piece values (for display)."""
        grid = np.zeros((8, 8), dtype=np.int8)
        for sq in range(32):
            r, c = SQ_TO_RC[sq]
            grid[r, c] = self.board[sq]
        return grid


# ---------------------------------------------------------------------------
# Game adapter
# ---------------------------------------------------------------------------

class CheckersGame(Game):
    """
    Checkers adapter for the AlphaZero plug-and-play interface.

    Informed by RaggedR/checkers (https://github.com/RaggedR/checkers).
    The source repo uses TD(lambda) with a hand-tuned 10-feature linear
    evaluator; this adapter wraps the same game engine for AlphaZero training
    with a learned neural evaluation function.
    """

    def get_display_name(self) -> str:
        return "Checkers"

    def get_action_size(self) -> int:
        return ACTION_SIZE  # 1024

    def get_init_state(self) -> CheckersState:
        return CheckersState()

    def get_next_state(
        self, state: CheckersState, player: int, action: int
    ) -> Tuple[CheckersState, int]:
        from_sq = action // 32
        to_sq   = action % 32
        # Find the matching move sequence for this (from, to) pair
        legal = state.get_legal_moves()
        move = self._find_move(legal, from_sq, to_sq)
        if move is None:
            raise ValueError(f"Illegal action {action} (from={from_sq}, to={to_sq})")
        next_state = state.apply_move(move)
        return next_state, next_state.turn

    def get_valid_moves(self, state: CheckersState, player: int) -> List[int]:
        legal = state.get_legal_moves()
        seen: Set[int] = set()
        actions: List[int] = []
        for move in legal:
            idx = move[0] * 32 + move[-1]
            if idx not in seen:
                seen.add(idx)
                actions.append(idx)
        return actions

    def get_game_ended(self, state: CheckersState, player: int) -> float:
        done, result = state.is_terminal()
        if not done:
            return 0
        # result is from the perspective of the player who just moved (state.turn was already flipped)
        # player is the one who just moved
        if abs(result) < 0.01:
            return 1e-4  # draw
        # If the current player (state.turn) has no moves, the previous player won.
        # The call semantics: player = the player whose move led to this state.
        return result

    def get_canonical_form(self, state: CheckersState, player: int) -> CheckersState:
        """
        Canonical form: always from the perspective of player +1 (black).
        If player is -1 (white), flip the board so white's pieces are encoded
        as +1/+2 and the board is 180-degree rotated.

        The correct rotation for the 32-square board is sq -> 31 - sq, which
        maps (row, col) -> (7-row, 7-col). A naive (7-row, col) flip doesn't
        work because even/odd rows have dark squares on different columns.
        """
        if player == 1:
            return state
        s = state.copy()
        # 180-degree rotation + negate pieces (swap black/white)
        new_board = [0] * 32
        for sq in range(32):
            new_board[31 - sq] = -state.board[sq]
        s.board = new_board
        s.turn = -state.turn
        return s

    def get_state_representation(self, state: CheckersState) -> np.ndarray:
        """
        Encode state as (4, 8, 8) tensor.

        Channels (canonical form — own = black/+1):
          0: own men   (value == 1)
          1: own kings (value == 2)
          2: opp men   (value == -1)
          3: opp kings (value == -2)
        """
        planes = np.zeros((4, 8, 8), dtype=np.float32)
        for sq in range(32):
            piece = state.board[sq]
            r, c = SQ_TO_RC[sq]
            if piece == 1:
                planes[0, r, c] = 1.0
            elif piece == 2:
                planes[1, r, c] = 1.0
            elif piece == -1:
                planes[2, r, c] = 1.0
            elif piece == -2:
                planes[3, r, c] = 1.0
        return planes

    def map_canonical_action(self, action: int, player: int) -> int:
        """
        Map a canonical-space action back to raw-space for the given player.

        For player +1 (black), canonical = raw, so identity.
        For player -1 (white), canonical form uses 180-degree rotation
        (sq -> 31-sq), so we reverse the mapping on both from and to squares.
        """
        if player == 1:
            return action
        from_sq = action // 32
        to_sq = action % 32
        return (31 - from_sq) * 32 + (31 - to_sq)

    def action_to_index(self, action: int) -> int:
        return action

    def index_to_action(self, index: int) -> int:
        return index

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_move(
        legal: List[List[int]], from_sq: int, to_sq: int
    ) -> Optional[List[int]]:
        """Find the first legal move matching (from_sq, to_sq)."""
        for move in legal:
            if move[0] == from_sq and move[-1] == to_sq:
                return move
        return None

    def get_symmetries(
        self, state: CheckersState, pi: np.ndarray
    ) -> List[Tuple[CheckersState, np.ndarray]]:
        """
        Checkers has left-right symmetry. Return the mirrored board + remapped policy.

        TODO: implement column-flip and remap action indices accordingly.
              For now returns [(state, pi)] only (no augmentation).
        """
        # TODO: implement horizontal flip symmetry for data augmentation
        return [(state, pi)]
