from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Any, List, Set, Sequence, Dict

from envs.tabular_matrix_env import TabularMatrixEnv, TabularParams, TabularState


class TreeMRPExactValue:
    """
    Exact evaluator for a binary tree Markov Reward Process (MRP) modeling
    chain-of-thought reasoning in RLVR (Reinforcement Learning with Verifiable Rewards).

    Structure:
      - Complete binary tree of depth `depth` with branching factor 2.
      - Total non-terminal tree nodes: 2^(depth + 1) - 1.
      - Absorbing terminal state: terminal_idx = 2^(depth + 1) - 1.
      - Start state: root (idx 0), corresponding to empty history "".
      - Actions: 1 (MRP, transitions are determined by environment dynamics).

    Branching dynamics:
      - At any intermediate node s:
        - 0 = left branch, 1 = right branch.
        - If both branches contain reachable correct (+1) leaves:
            P(left) = p_both (default 0.5), P(right) = 1.0 - p_both (0.5).
            (One branch chosen arbitrarily).
        - If only one branch contains reachable correct (+1) leaves:
            P(correct branch) = p, P(wrong branch) = 1.0 - p.
            (Mimics taking a key reasoning step with success probability p).
        - If neither branch contains reachable correct (+1) leaves:
            P(left) = p_neither (default 0.5), P(right) = 1.0 - p_neither (0.5).
      - At leaf nodes (depth == depth):
        - Transitions to terminal_idx with probability 1.0.
        - Yields reward R = 1.0 on exit if the leaf is in the correct cluster, else 0.0.
      - At terminal_idx:
        - Self-loops with probability 1.0 and reward 0.0.

    Observations:
      - Each state observation is a fixed-length numeric vector of shape (depth,)
        recording all past binary branches taken so far (e.g. [1.0, 1.0, 0.0, -1.0, -1.0]),
        padded with `pad_val` (-1.0) for future / unreached steps.
      - Also provides string representations for all states (e.g. "110", "<TERM>").
    """

    def __init__(
        self,
        depth: int = 5,
        p: float = 0.8,
        p_both: float = 0.5,
        p_neither: float = 0.5,
        gamma: float = 1.0,
        cluster_size: int | None = None,
        cluster_start_idx: int | None = None,
        correct_leaves: Sequence[int | str] | None = None,
        reward_correct: float = 1.0,
        reward_incorrect: float = 0.0,
        pad_val: float = -1.0,
    ):
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")

        self.depth = int(depth)
        self.p = float(p)
        self.p_both = float(p_both)
        self.p_neither = float(p_neither)
        self.gamma = float(gamma)
        self.reward_correct = float(reward_correct)
        self.reward_incorrect = float(reward_incorrect)
        self.pad_val = float(pad_val)

        # 1. State counts & indices
        # Level d has 2^d nodes. Total tree nodes = 2^(depth + 1) - 1
        self.num_states = (1 << (self.depth + 1)) - 1
        self.num_actions = 1  # MRP
        self.terminal_idx = self.num_states
        self.num_total_states = self.num_states + 1
        self.start_idx = 0
        self.goal_idx = self.terminal_idx  # Terminal state observation on exit

        # Leaves are nodes at depth == self.depth
        self.leaf_offset = (1 << self.depth) - 1
        self.num_leaves = 1 << self.depth
        self.leaf_indices = list(range(self.leaf_offset, self.leaf_offset + self.num_leaves))

        # Partition nodes by depth
        self.nodes_by_depth: Dict[int, List[int]] = {}
        for d in range(self.depth + 1):
            start = (1 << d) - 1
            count = 1 << d
            self.nodes_by_depth[d] = list(range(start, start + count))

        # 2. String representations & Coordinates [depth, offset_at_depth]
        self.state_strings: List[str] = ["" for _ in range(self.num_total_states)]
        coords_list = []

        for d in range(self.depth + 1):
            nodes = self.nodes_by_depth[d]
            for offset, s_idx in enumerate(nodes):
                coords_list.append([d, offset])
                if s_idx == 0:
                    self.state_strings[s_idx] = ""
                else:
                    parent = (s_idx - 1) // 2
                    branch_taken = "0" if (s_idx % 2 == 1) else "1"
                    self.state_strings[s_idx] = self.state_strings[parent] + branch_taken

        # Terminal state coordinates and string
        coords_list.append([self.depth + 1, 0])
        self.state_strings[self.terminal_idx] = "<TERM>"

        self.coords = jnp.array(coords_list, dtype=jnp.int32)
        self.start = self.coords[self.start_idx]
        self.goal = self.coords[self.goal_idx]
        self.occupied_map = jnp.zeros((1, 1), dtype=jnp.float32)
        self.use_visual_obs = False

        # 3. Determine Correct Leaves (+1 reward cluster)
        self.correct_leaf_indices: Set[int] = self._resolve_correct_leaves(
            cluster_size=cluster_size,
            cluster_start_idx=cluster_start_idx,
            correct_leaves=correct_leaves,
        )

        # 4. Count correct leaves reachable from each node (bottom-up)
        self.num_correct_reachable = self._compute_correct_reachability()

        # 5. Build Observations stack
        self.obs_stack = self._build_obs_stack()

        # 6. Build Dynamics P and R
        self.P, self.R = self._build_env_dynamics()

    def _resolve_correct_leaves(
        self,
        cluster_size: int | None,
        cluster_start_idx: int | None,
        correct_leaves: Sequence[int | str] | None,
    ) -> Set[int]:
        """Resolves the set of global state indices corresponding to correct (+1) leaves."""
        if correct_leaves is not None:
            resolved = set()
            for item in correct_leaves:
                if isinstance(item, str):
                    # Interpret as binary path string of length self.depth
                    s_idx = self.string_to_idx(item)
                    if s_idx not in self.leaf_indices:
                        raise ValueError(f"Binary string '{item}' does not correspond to a leaf at depth {self.depth}")
                    resolved.add(s_idx)
                elif isinstance(item, (int, np.integer)):
                    # Can be local leaf index 0..num_leaves-1 or global state index
                    val = int(item)
                    if 0 <= val < self.num_leaves:
                        resolved.add(self.leaf_offset + val)
                    elif self.leaf_offset <= val < self.terminal_idx:
                        resolved.add(val)
                    else:
                        raise ValueError(f"Leaf index {val} out of bounds")
                else:
                    raise TypeError(f"Unsupported leaf identifier: {item}")
            return resolved

        # Default: contiguous cluster of leaves
        if cluster_size is None:
            # Default ~10% of leaves or at least 1-2 leaves
            cluster_size = max(1, 1 << max(0, self.depth - 3))
        cluster_size = max(1, min(cluster_size, self.num_leaves))

        if cluster_start_idx is None:
            # Place cluster towards the right (e.g. all ones branch)
            cluster_start_idx = self.num_leaves - cluster_size
        cluster_start_idx = max(0, min(cluster_start_idx, self.num_leaves - cluster_size))

        correct_set = set()
        for k in range(cluster_start_idx, cluster_start_idx + cluster_size):
            correct_set.add(self.leaf_offset + k)
        return correct_set

    def _compute_correct_reachability(self) -> np.ndarray:
        """Bottom-up computation of number of correct leaves reachable from each node."""
        reach = np.zeros(self.num_total_states, dtype=np.int32)
        for leaf in self.leaf_indices:
            reach[leaf] = 1 if leaf in self.correct_leaf_indices else 0

        # Traverse backwards from depth-1 down to 0
        for d in range(self.depth - 1, -1, -1):
            for s in self.nodes_by_depth[d]:
                left = 2 * s + 1
                right = 2 * s + 2
                reach[s] = reach[left] + reach[right]

        return reach

    def _build_obs_stack(self) -> jax.Array:
        """
        Builds the observation vector for each state of shape (num_total_states, depth).
        Prefix of length d contains binary choices (0.0 or 1.0).
        Remaining (depth - d) entries are padded with self.pad_val (-1.0).
        """
        obs = np.full((self.num_total_states, self.depth), self.pad_val, dtype=np.float32)

        for s in range(self.num_states):
            s_str = self.state_strings[s]
            for i, char in enumerate(s_str):
                obs[s, i] = 1.0 if char == "1" else 0.0

        # Terminal state: padded vector
        obs[self.terminal_idx, :] = self.pad_val
        return jnp.asarray(obs, dtype=jnp.float32)

    def _build_env_dynamics(self) -> Tuple[jax.Array, jax.Array]:
        """
        Builds transition tensor P and reward tensor R.
        P: (num_total_states, num_actions, num_total_states)
        R: (num_total_states, num_actions, num_total_states)
        """
        P = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)
        R = np.zeros((self.num_total_states, self.num_actions, self.num_total_states), dtype=np.float32)

        # 1. Internal tree nodes (depth < self.depth)
        for d in range(self.depth):
            for s in self.nodes_by_depth[d]:
                left = 2 * s + 1
                right = 2 * s + 2
                c_left = self.num_correct_reachable[left]
                c_right = self.num_correct_reachable[right]

                if c_left > 0 and c_right > 0:
                    # Both branches lead to correct leaves: chosen arbitrarily
                    p_left = self.p_both
                    p_right = 1.0 - self.p_both
                elif c_left > 0 and c_right == 0:
                    # Left branch leads to correct leaf
                    p_left = self.p
                    p_right = 1.0 - self.p
                elif c_left == 0 and c_right > 0:
                    # Right branch leads to correct leaf
                    p_left = 1.0 - self.p
                    p_right = self.p
                else:
                    # Neither branch leads to correct leaf (dead end)
                    p_left = self.p_neither
                    p_right = 1.0 - self.p_neither

                P[s, 0, left] = p_left
                P[s, 0, right] = p_right
                R[s, 0, :] = 0.0

        # 2. Leaf nodes (depth == self.depth) -> exit to terminal_idx
        for leaf in self.leaf_indices:
            P[leaf, 0, self.terminal_idx] = 1.0
            r_leaf = self.reward_correct if leaf in self.correct_leaf_indices else self.reward_incorrect
            R[leaf, 0, self.terminal_idx] = r_leaf

        # 3. Terminal state: absorbing self-loop with reward 0.0
        P[self.terminal_idx, 0, self.terminal_idx] = 1.0
        R[self.terminal_idx, 0, self.terminal_idx] = 0.0

        return jnp.asarray(P, dtype=jnp.float32), jnp.asarray(R, dtype=jnp.float32)

    def solve_backward_induction(self) -> jax.Array:
        """
        Exact single-pass backward induction from leaves to root in O(num_states) time.
        No matrix inversion needed because the tree has no cycles and known finite depth.
        """
        V = np.zeros(self.num_total_states, dtype=np.float32)
        V[self.terminal_idx] = 0.0

        # 1. Leaves: V(leaf) = R(leaf -> terminal)
        for leaf in self.leaf_indices:
            V[leaf] = float(self.R[leaf, 0, self.terminal_idx])

        # 2. Internal nodes from depth - 1 down to 0
        P_np = np.asarray(self.P)
        for d in range(self.depth - 1, -1, -1):
            for s in self.nodes_by_depth[d]:
                left = 2 * s + 1
                right = 2 * s + 2
                V[s] = self.gamma * (P_np[s, 0, left] * V[left] + P_np[s, 0, right] * V[right])

        return jnp.asarray(V, dtype=jnp.float32)

    def solve_linear_system(self, pi: jax.Array | None = None) -> jax.Array:
        """
        Solves the Bellman equation (I - gamma * P_pi) V = R_pi via standard linear solve.
        For verification against solve_backward_induction().
        """
        if pi is None:
            # Single action MRP: deterministic policy on action 0
            pi = jnp.ones((self.num_total_states, self.num_actions), dtype=jnp.float32)

        P_pi = jnp.einsum("sa,sam->sm", pi, self.P)
        R_pi = jnp.einsum("sa,sam,sam->s", pi, self.P, self.R)

        A = jnp.eye(self.num_total_states) - self.gamma * P_pi
        # For the absorbing terminal state with V=0 and R=0:
        # A[terminal, terminal] = 1.0 - gamma, which can be singular if gamma = 1.0.
        # We explicitly enforce V[terminal_idx] = 0.0.
        A = A.at[self.terminal_idx, :].set(0.0)
        A = A.at[self.terminal_idx, self.terminal_idx].set(1.0)
        R_pi = R_pi.at[self.terminal_idx].set(0.0)

        return jnp.linalg.solve(A, R_pi)

    def compute_true_values_raw(self, pi: jax.Array | None = None) -> jax.Array:
        """Returns exact value vector of shape (num_total_states,)."""
        return self.solve_backward_induction()

    def compute_true_values(self, pi: jax.Array | None = None) -> jax.Array:
        """Returns exact value vector of shape (num_total_states,)."""
        return self.solve_backward_induction()

    # --- Helper Conversion Methods ---

    def string_to_idx(self, path_str: str) -> int:
        """Converts a binary path string (e.g. '1101') to its global state index."""
        if path_str == "<TERM>" or path_str == "DONE":
            return self.terminal_idx
        d = len(path_str)
        if d > self.depth:
            raise ValueError(f"String length {d} exceeds tree depth {self.depth}")
        if d == 0:
            return 0
        offset = int(path_str, 2)
        start_of_level = (1 << d) - 1
        return start_of_level + offset

    def idx_to_string(self, s_idx: int) -> str:
        """Converts a global state index to its binary path string."""
        return self.state_strings[int(s_idx)]

    def idx_to_obs(self, s_idx: int) -> np.ndarray:
        """Returns the float32 observation vector for state s_idx."""
        return np.asarray(self.obs_stack[int(s_idx)])

    def format_tree_values(self, V: jax.Array | np.ndarray | None = None) -> str:
        """Returns a formatted multi-line string displaying the values along the tree."""
        if V is None:
            V = self.solve_backward_induction()
        V_arr = np.asarray(V)

        lines = [f"TreeMRP (depth={self.depth}, p={self.p}, gamma={self.gamma}):"]
        for d in range(self.depth + 1):
            nodes = self.nodes_by_depth[d]
            lines.append(f"\n--- Depth {d} ({len(nodes)} nodes) ---")
            for s in nodes:
                s_str = self.state_strings[s]
                disp_str = f'"{s_str}"' if s_str != "" else '"" (Root)'
                val = V_arr[s]
                extra = ""
                if d == self.depth:
                    is_correct = s in self.correct_leaf_indices
                    extra = f" [CORRECT LEAF: R={self.reward_correct}]" if is_correct else " [R=0]"
                else:
                    reach = self.num_correct_reachable[s]
                    extra = f" (correct leaves reachable: {reach})"
                lines.append(f"  State {s:3d} | Path {disp_str:16s} | V = {val:8.4f}{extra}")

        lines.append(f"\nTerminal State {self.terminal_idx}: V = {V_arr[self.terminal_idx]:.4f}")
        return "\n".join(lines)


class TreeMRP(TabularMatrixEnv):
    """
    Gymnax-compatible matrix environment for the Tree MRP,
    simulated directly via TabularMatrixEnv.
    """

    def __init__(
        self,
        depth: int = 5,
        p: float = 0.8,
        gamma: float = 1.0,
        cluster_size: int | None = None,
        cluster_start_idx: int | None = None,
        correct_leaves: Sequence[int | str] | None = None,
        name: str | None = None,
        **kwargs,
    ):
        evaluator = TreeMRPExactValue(
            depth=depth,
            p=p,
            gamma=gamma,
            cluster_size=cluster_size,
            cluster_start_idx=cluster_start_idx,
            correct_leaves=correct_leaves,
            **kwargs,
        )
        env_name = name or f"TreeMRP-d{depth}-p{p}"
        super().__init__(evaluator, name=env_name)
