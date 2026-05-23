"""

    Tetris environment.

"""

import numpy as np
import random

# numpy random seed
np.random.seed(0)

# Tetromino shapes
TETROMINOES = {

    'I': [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],

    'O': [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],

    'T': [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (1, -1)],
    ],

    'S': [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],

    'Z': [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],

    'J': [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, -1)],
    ],

    'L': [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],

}

PIECE_NAMES = list(TETROMINOES.keys())

COLORS = {
    'I': (0, 255, 255),
    'O': (255, 255, 0),
    'T': (128, 0, 128),
    'S': (0, 255, 0),
    'Z': (255, 0, 0),
    'J': (0, 0, 255),
    'L': (255, 165, 0),
}


class TetrisEnv:

    """

    Tetris environment.

    1. **manual**: Slow progress.

    2. **training mode**: Instant hard-drop.

    """

    def __init__(self, rows=20, cols=10):

        self.rows = rows
        self.cols = cols
        self._init_geometry_cache()

        self.reset()

    def _init_geometry_cache(self):

        self._piece_rotations = {
            piece: [np.array(shape, dtype=np.int16) for shape in rotations]
            for piece, rotations in TETROMINOES.items()
        }
        self._shape_bounds = {
            piece: [
                (
                    int(shape[:, 0].min()),
                    int(shape[:, 0].max()),
                    int(shape[:, 1].min()),
                    int(shape[:, 1].max()),
                )
                for shape in rotations
            ]
            for piece, rotations in self._piece_rotations.items()
        }

        self._row_from_bottom = (self.rows - 1 - np.arange(self.rows, dtype=np.float32))[:, None]

        if self.cols == 1:
            self._col_positions = np.zeros((1, self.cols), dtype=np.float32)
        else:
            self._col_positions = np.linspace(0.0, 1.0, self.cols, dtype=np.float32)[None, :]

        self._distance_from_middle = np.abs(1.0 - 2.0 * self._col_positions).astype(np.float32)
        self._feature_names = self._build_state_feature_names()


    def reset(self):

        self.board = np.zeros( (self.rows, self.cols), dtype=np.float32)

        # init variables
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.current_piece = self._random_piece()
        self.next_piece = self._random_piece()

        # manual play state
        self.piece_row = 0
        self.piece_col = 0
        self.piece_rotation = 0
        self.piece_active = False

        return self._get_state()

    def _random_piece(self):

        return random.choice(PIECE_NAMES)

    def _get_rotations(self, piece_name):

        return self._piece_rotations[piece_name]

    def _current_shape(self):

        """
        Return the block offsets for the current rotation.
        """

        return self._get_rotations(self.current_piece)[self.piece_rotation]


    def _is_valid_position(self, shape, row, col):

        """
        Check if placing shape at (row, col) is inside bounds and empty.
        """

        rows = row + shape[:, 0]
        cols = col + shape[:, 1]

        if (
            np.any(rows < 0)
            or np.any(rows >= self.rows)
            or np.any(cols < 0)
            or np.any(cols >= self.cols)
        ):
            return False
            
        return not np.any(self.board[rows, cols] > 0)

    # manual play methods

    def spawn_piece(self):

        """
            spawn the current piece at the top-center of the board
            returns False if the piece cannot be spawned (game over)
        """
        
        self.piece_rotation = 0

        shape = self._current_shape()
        _, _, min_col, max_col = self._shape_bounds[self.current_piece][self.piece_rotation]

        self.piece_col = (self.cols - (max_col - min_col + 1)) // 2 - min_col
        self.piece_row = 0

        if not self._is_valid_position(shape, self.piece_row, self.piece_col):

            self.game_over = True
            self.piece_active = False

            return False

        self.piece_active = True
        return True

    def move_left(self):
        if not self.piece_active:
            return False
        shape = self._current_shape()
        if self._is_valid_position(shape, self.piece_row, self.piece_col - 1):
            self.piece_col -= 1
            return True
        return False

    def move_right(self):
        if not self.piece_active:
            return False
        shape = self._current_shape()
        if self._is_valid_position(shape, self.piece_row, self.piece_col + 1):
            self.piece_col += 1
            return True
        return False

    def rotate(self):
        """Rotate the piece clockwise. Tries the new rotation; reverts if invalid."""
        if not self.piece_active:
            return False
        rotations = self._get_rotations(self.current_piece)
        new_rot = (self.piece_rotation + 1) % len(rotations)
        new_shape = rotations[new_rot]
        if self._is_valid_position(new_shape, self.piece_row, self.piece_col):
            self.piece_rotation = new_rot
            return True
        # Simple wall-kick: try shifting left or right by 1
        for nudge in [-1, 1, -2, 2]:
            if self._is_valid_position(new_shape, self.piece_row, self.piece_col + nudge):
                self.piece_col += nudge
                self.piece_rotation = new_rot
                return True
        return False

    def tick(self):
        """
        Move the active piece down by one row (gravity).
        If it can't move down, lock it in place, clear lines, and advance.
        Returns: (reward, done, info)
        """
        if not self.piece_active:
            return 0, self.game_over, {"score": self.score, "lines_cleared": self.lines_cleared}

        shape = self._current_shape()

        if self._is_valid_position(shape, self.piece_row + 1, self.piece_col):
            self.piece_row += 1
            return 0, False, {"score": self.score, "lines_cleared": self.lines_cleared}
        else:
            return self._lock_and_advance()

    def soft_drop(self):
        """Move piece down one row immediately (player pressed down)."""
        return self.tick()

    def hard_drop_live(self):
        """
        Instantly drop the active piece to the lowest valid row and lock it.
        Returns: (reward, done, info)
        """
        if not self.piece_active:
            return 0, self.game_over, {"score": self.score, "lines_cleared": self.lines_cleared}

        shape = self._current_shape()
        drop_row = self._get_drop_row(shape, self.piece_col, start_row=self.piece_row)

        if drop_row >= self.piece_row:
            self.piece_row = drop_row

        return self._lock_and_advance()

    def _lock_and_advance(self):
        """Lock the current piece onto the board, clear lines, spawn next."""
        shape = self._current_shape()
        self._place_shape_on_board(self.board, shape, self.piece_row, self.piece_col)

        self.piece_active = False
        cleared = self._clear_lines()

        line_rewards = {
            0: 0, 
            1: 40, 
            2: 100, 
            3: 300, 
            4: 1200
        }

        reward = line_rewards.get(cleared, 1200)

        self.score += reward
        self.lines_cleared += cleared

        # Advance to next piece
        self.current_piece = self.next_piece
        self.next_piece = self._random_piece()

        if not self.spawn_piece():
            reward -= 10
            return reward, True, {"score": self.score, "lines_cleared": self.lines_cleared}

        return reward, False, {"score": self.score, "lines_cleared": self.lines_cleared}

    def get_ghost_row(self):

        """
            return the row where the piece would land if hard-dropped (for preview).
        """

        if not self.piece_active:
            return self.piece_row
        
        shape = self._current_shape()
        drop_row = self._get_drop_row(shape, self.piece_col, start_row=self.piece_row)

        return max(self.piece_row, drop_row)

    # line clearing
    def _clear_lines(self):

        self.board, cleared = self._clear_lines_from_board(self.board)

        return cleared

    def _clear_lines_from_board(self, board):

        full_rows = np.all(board > 0, axis=1)
        cleared = int(np.sum(full_rows))

        if cleared == 0:
            return board, 0

        empty_rows = np.zeros((cleared, self.cols), dtype=np.float32)
        remaining_rows = board[~full_rows]

        return np.vstack([empty_rows, remaining_rows]).astype(np.float32), cleared

    # RL training interface (instant hard-drop)

    def get_possible_actions(self):

        """
            Returns a list of (rotation_index, column) tuples for all valid
            placements of the current piece (used by the RL agent).
        """

        actions = []
        rotations = self._get_rotations(self.current_piece)

        for rot_idx, shape in enumerate(rotations):

            _, _, min_col, max_col = self._shape_bounds[self.current_piece][rot_idx]
            for col in range(-min_col, self.cols - max_col):
                actions.append((rot_idx, col))

        return actions

    def _check_hard_drop_valid(self, shape, col):

        cols = col + shape[:, 1]

        return bool(np.all((cols >= 0) & (cols < self.cols)))

    def _get_drop_row(self, shape, col, start_row=0, board=None):

        if board is None:
            board = self.board

        cols = col + shape[:, 1]
        if np.any(cols < 0) or np.any(cols >= self.cols):
            return -1

        max_start_row = self.rows - 1 - int(np.max(shape[:, 0]))
        if start_row > max_start_row:
            return -1

        candidate_rows = np.arange(start_row, max_start_row + 1, dtype=np.int16)
        board_rows = candidate_rows[:, None] + shape[:, 0][None, :]
        occupied = board[board_rows, cols[None, :]] > 0
        valid_rows = ~np.any(occupied, axis=1)

        if valid_rows.size == 0 or not valid_rows[0]:
            return -1 if start_row == 0 else start_row

        first_invalid = np.flatnonzero(~valid_rows)
        if first_invalid.size == 0:
            return int(candidate_rows[-1])

        return int(candidate_rows[first_invalid[0] - 1])

    def _place_shape_on_board(self, board, shape, row, col):

        rows = row + shape[:, 0]
        cols = col + shape[:, 1]
        valid_cells = (
            (rows >= 0)
            & (rows < self.rows)
            & (cols >= 0)
            & (cols < self.cols)
        )

        board[rows[valid_cells], cols[valid_cells]] = 1.0

    def step(self, action):

        """
            RL training step: instantly hard-drop the piece at (rotation, column)
            returns: (state, reward, done, info)
        """

        rot_idx, col = action

        shape = self._get_rotations(self.current_piece)[rot_idx]

        drop_row = self._get_drop_row(shape, col)

        if drop_row < 0:

            self.game_over = True

            return self._get_state(), -10.0, True, {"score": self.score}

        # lock piece in place
        self._place_shape_on_board(self.board, shape, drop_row, col)

        cleared = self._clear_lines()

        line_rewards = {
            0: 0, 
            1: 40, 
            2: 100, 
            3: 300, 
            4: 1200
        }

        reward = line_rewards.get(cleared, 1200) # upper bound for more than 4 lines cleared

        self.score += reward
        self.lines_cleared += cleared

        max_height = float(np.max(self._column_heights_from_board(self.board)))
        reward -= max_height * 0.5

        self.current_piece = self.next_piece
        self.next_piece = self._random_piece()

        if not self.get_possible_actions():

            self.game_over = True
            reward -= 10.0

        info_ = {
            "score": self.score,
            "lines_cleared": self.lines_cleared,
        }

        return self._get_state(), reward, self.game_over, info_

    # state features for the RL agent

    def _column_heights_from_board(self, board):

        filled = board > 0
        has_block = np.any(filled, axis=0)
        first_filled_rows = np.argmax(filled, axis=0)

        return np.where(has_block, self.rows - first_filled_rows, 0).astype(np.float32)

    def _hole_mask_from_board(self, board):

        filled = board > 0
        inside_column_height = np.maximum.accumulate(filled, axis=0)

        return inside_column_height & ~filled

    def _masked_mean(self, values, mask, axis):

        counts = np.sum(mask, axis=axis).astype(np.float32)
        sums = np.sum(np.where(mask, values, 0.0), axis=axis).astype(np.float32)

        return np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=np.float32),
            where=counts > 0,
        )

    def _masked_std(self, values, mask, axis, min_count=2):

        counts = np.sum(mask, axis=axis).astype(np.float32)
        means = self._masked_mean(values, mask, axis)
        expanded_means = np.expand_dims(means, axis=axis)
        squared_error = np.where(mask, (values - expanded_means) ** 2, 0.0)
        variance_sums = np.sum(squared_error, axis=axis).astype(np.float32)

        variances = np.divide(
            variance_sums,
            counts,
            out=np.zeros_like(variance_sums, dtype=np.float32),
            where=counts >= min_count,
        )

        return np.sqrt(variances).astype(np.float32)

    def _masked_mean_scalar(self, values, mask):

        count = float(np.sum(mask))
        if count == 0:
            return 0.0

        return float(np.sum(np.where(mask, values, 0.0)) / count)

    def _masked_std_scalar(self, values, mask, min_count=2):

        count = float(np.sum(mask))
        if count < min_count:
            return 0.0

        mean = self._masked_mean_scalar(values, mask)
        variance = np.sum(np.where(mask, (values - mean) ** 2, 0.0)) / count

        return float(np.sqrt(variance))

    def _safe_group_mean(self, values, valid_groups):

        if not np.any(valid_groups):
            return 0.0

        return float(np.mean(values[valid_groups]))

    def _safe_group_std(self, values, valid_groups):

        if not np.any(valid_groups):
            return 0.0

        return float(np.std(values[valid_groups]))

    def _clusteredness_from_std(self, std_values, valid_groups):

        clusteredness = 1.0 - 2.0 * std_values
        clusteredness = np.clip(clusteredness, 0.0, 1.0).astype(np.float32)

        return np.where(valid_groups, clusteredness, 0.0).astype(np.float32)

    def _get_column_heights(self):

        return self._column_heights_from_board(self.board).astype(int).tolist()

    def _count_holes(self):

        return int(np.sum(self._hole_mask_from_board(self.board)))

    def _bumpiness(self):

        heights = self._column_heights_from_board(self.board)

        return float(np.sum(np.abs(np.diff(heights))))

    def _complete_lines(self):

        return int(np.sum(np.all(self.board > 0, axis=1)))

    def _get_board_features(self, board):

        heights = self._column_heights_from_board(board)
        fill_heights = heights / float(self.rows)

        total_height = float(np.sum(heights))
        total_holes_denominator = total_height if total_height > 0.0 else 1.0

        mean_height = float(np.mean(fill_heights))
        height_deviation = float(np.std(fill_heights))
        lowest_point = float(np.min(fill_heights))
        highest_point = float(np.max(fill_heights))

        hole_mask = self._hole_mask_from_board(board)
        hole_counts_per_col = np.sum(hole_mask, axis=0).astype(np.float32)
        hole_counts_per_row = np.sum(hole_mask, axis=1).astype(np.float32)
        total_holes = float(np.sum(hole_counts_per_col))
        total_holeyness = total_holes / total_holes_denominator

        hole_height_per_col = np.divide(
            hole_counts_per_col,
            heights,
            out=np.zeros_like(hole_counts_per_col, dtype=np.float32),
            where=heights > 0,
        )

        height_denominator = np.maximum(heights - 1.0, 1.0)[None, :]
        hole_heights = self._row_from_bottom / height_denominator
        vertical_depths = 1.0 - hole_heights

        vertical_hole_depths = self._masked_mean(vertical_depths, hole_mask, axis=0)
        vertical_hole_height_std = self._masked_std(hole_heights, hole_mask, axis=0)
        valid_vertical_clusters = hole_counts_per_col >= 2
        vertical_hole_clusteredness = self._clusteredness_from_std(
            vertical_hole_height_std,
            valid_vertical_clusters,
        )

        horizontal_hole_distances = self._masked_mean(self._distance_from_middle, hole_mask, axis=1)
        horizontal_hole_position_std = self._masked_std(self._col_positions, hole_mask, axis=1)
        valid_horizontal_clusters = hole_counts_per_row >= 2
        horizontal_hole_clusteredness = self._clusteredness_from_std(
            horizontal_hole_position_std,
            valid_horizontal_clusters,
        )

        valid_vertical_depths = hole_counts_per_col >= 1
        mean_hole_depth = self._safe_group_mean(vertical_hole_depths, valid_vertical_depths)
        hole_depth_deviation = self._safe_group_std(vertical_hole_depths, valid_vertical_depths)
        mean_hole_vertical_clusteredness = self._safe_group_mean(
            vertical_hole_clusteredness,
            valid_vertical_clusters,
        )
        hole_vertical_instability = self._safe_group_std(
            vertical_hole_clusteredness,
            valid_vertical_clusters,
        )

        valid_horizontal_distances = hole_counts_per_row >= 1
        mean_hole_edge_distance = self._safe_group_mean(
            horizontal_hole_distances,
            valid_horizontal_distances,
        )
        hole_edge_distance_deviation = self._safe_group_std(
            horizontal_hole_distances,
            valid_horizontal_distances,
        )
        mean_hole_horizontal_clusteredness = self._safe_group_mean(
            horizontal_hole_clusteredness,
            valid_horizontal_clusters,
        )
        hole_horizontal_instability = self._safe_group_std(
            horizontal_hole_clusteredness,
            valid_horizontal_clusters,
        )

        hole_distances = np.sqrt((vertical_depths ** 2 + self._distance_from_middle ** 2) / 2.0)
        mean_hole_distance = self._masked_mean_scalar(hole_distances, hole_mask)

        # True 2D clusteredness: spread of hole coordinates, not spread of a scalar distance.
        hole_x_std = self._masked_std_scalar(self._col_positions, hole_mask)
        hole_y_std = self._masked_std_scalar(vertical_depths, hole_mask)
        max_2d_std = np.sqrt(0.5)
        general_hole_clusteredness = 0.0
        if total_holes >= 2.0:
            general_hole_clusteredness = 1.0 - (np.sqrt(hole_x_std ** 2 + hole_y_std ** 2) / max_2d_std)
            general_hole_clusteredness = float(np.clip(general_hole_clusteredness, 0.0, 1.0))

        features = np.concatenate([
            fill_heights,
            np.array([
                total_holeyness,
                mean_height,
                height_deviation,
                lowest_point,
                highest_point,
            ], dtype=np.float32),
            hole_height_per_col,
            vertical_hole_depths,
            vertical_hole_clusteredness,
            horizontal_hole_distances,
            horizontal_hole_clusteredness,
            np.array([
                mean_hole_depth,
                mean_hole_vertical_clusteredness,
                hole_depth_deviation,
                hole_vertical_instability,
                mean_hole_edge_distance,
                mean_hole_horizontal_clusteredness,
                hole_edge_distance_deviation,
                hole_horizontal_instability,
                mean_hole_distance,
                general_hole_clusteredness,
            ], dtype=np.float32),
        ])

        return features.astype(np.float32)

    def _get_state(self):

        return self._get_board_features(self.board)

    def get_state_for_action(self, action):

        """
            simulate an action and return the resulting state (without modifying the board)
        """

        rot_idx, col = action
        shape = self._get_rotations(self.current_piece)[rot_idx]
        board_copy = self.board.copy()
        drop_row = self._get_drop_row(shape, col)

        if drop_row < 0:
            return None
        
        self._place_shape_on_board(board_copy, shape, drop_row, col)

        board_copy, _ = self._clear_lines_from_board(board_copy)

        return self._get_board_features(board_copy)

    def _build_state_feature_names(self):

        return (
            [f"fill_height_col_{c}" for c in range(self.cols)]
            + [
                "total_holeyness",
                "mean_height",
                "height_deviation",
                "lowest_point",
                "highest_point",
            ]
            + [f"holeyness_col_{c}" for c in range(self.cols)]
            + [f"vertical_hole_depth_col_{c}" for c in range(self.cols)]
            + [f"vertical_hole_clusteredness_col_{c}" for c in range(self.cols)]
            + [f"horizontal_hole_edge_distance_row_{r}" for r in range(self.rows)]
            + [f"horizontal_hole_clusteredness_row_{r}" for r in range(self.rows)]
            + [
                "mean_hole_depth",
                "mean_hole_vertical_clusteredness",
                "hole_depth_deviation",
                "hole_vertical_instability",
                "mean_hole_edge_distance",
                "mean_hole_horizontal_clusteredness",
                "hole_edge_distance_deviation",
                "hole_horizontal_instability",
                "mean_hole_distance",
                "general_hole_clusteredness",
            ]
        )

    @property
    def state_feature_names(self):

        return self._feature_names

    @property
    def state_size(self):
        return len(self._feature_names)
