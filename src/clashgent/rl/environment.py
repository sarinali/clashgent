"""Gymnasium environment wrapper for Clash Royale."""

import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..bridges.base import EmulatorBridge
from ..game.actions import ActionSpace, GameAction
from ..game.state import GameState
from ..verifiers.base import Verifier
from ..vision.extractor import StateExtractor


class ClashEnv(gym.Env):
    """Gymnasium environment wrapper for Clash Royale.

    This environment bridges the RL agent with the actual game running
    in an emulator. It handles:
    - State observation via screenshot capture and vision processing
    - Action execution via emulator input
    - Reward calculation with optional verifier shaping

    Attributes:
        bridge: Emulator bridge for screenshots and actions
        state_extractor: Vision system for extracting game state
        verifiers: List of reward shaping verifiers
        frame_skip: Number of frames to skip between actions
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        bridge: EmulatorBridge,
        state_extractor: StateExtractor,
        verifiers: Optional[list[Verifier]] = None,
        frame_skip: int = 4,
        action_delay: float = 0.1,
        obs_dim: int = 256,
    ):
        """Initialize Clash Royale environment.

        Args:
            bridge: Emulator bridge for screenshots and actions
            state_extractor: Vision system for game state extraction
            verifiers: Optional reward shaping verifiers
            frame_skip: Frames to skip between agent decisions
            action_delay: Delay in seconds between actions
            obs_dim: Dimension of encoded observation vector
        """
        super().__init__()

        self.bridge = bridge
        self.state_extractor = state_extractor
        self.verifiers = verifiers or []
        self.frame_skip = frame_skip
        self.action_delay = action_delay
        self.obs_dim = obs_dim

        self.action_space_def = ActionSpace()

        # Define observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Define action space
        self.action_space = spaces.Discrete(self.action_space_def.num_actions)

        # Internal state
        self._current_state: Optional[GameState] = None
        self._prev_state: Optional[GameState] = None
        self._episode_start_time: float = 0
        self._step_count: int = 0
        self._last_screenshot: Optional[np.ndarray] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment for new episode.

        This should be called at the start of each match. The implementation
        should handle:
        - Navigating to start a new match (if needed)
        - Waiting for match to begin
        - Capturing initial game state

        Args:
            seed: Random seed for reproducibility
            options: Optional configuration dict

        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)

        # Reset verifiers
        for verifier in self.verifiers:
            verifier.reset()

        # TODO: Implement match start navigation
        # This might involve:
        # - Detecting if in menu vs match
        # - Clicking "Battle" button
        # - Waiting for matchmaking
        # - Detecting match start

        self._episode_start_time = time.time()
        self._step_count = 0

        # Capture initial state
        screenshot = self.bridge.capture()
        self._last_screenshot = screenshot
        self._current_state = self.state_extractor.extract(screenshot)
        self._prev_state = None

        obs = self._encode_state(self._current_state)
        info = {
            "game_state": self._current_state,
            "step": self._step_count,
        }

        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute action and return new state.

        Args:
            action: Discrete action index

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._prev_state = self._current_state
        self._step_count += 1

        # Convert discrete action to game action
        game_action = self.action_space_def.to_discrete_action(action)

        # Execute action
        success = self.bridge.execute(game_action)

        # Frame skip - wait for game to progress
        time.sleep(self.action_delay * self.frame_skip)

        # Capture new state
        screenshot = self.bridge.capture()
        self._last_screenshot = screenshot
        self._current_state = self.state_extractor.extract(screenshot)

        # Calculate reward
        reward = self._calculate_reward(game_action)

        # Check termination
        terminated = self._is_match_over()
        truncated = self._is_truncated()

        obs = self._encode_state(self._current_state)
        info = {
            "game_state": self._current_state,
            "prev_state": self._prev_state,
            "action": game_action,
            "action_success": success,
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def _encode_state(self, state: GameState) -> np.ndarray:
        """Encode GameState to observation vector.

        Converts the structured game state into a fixed-size vector
        suitable for neural network input.

        Args:
            state: Current game state

        Returns:
            Normalized observation vector
        """
        # TODO: Implement proper state encoding
        # This should include:
        # - Elixir (normalized 0-1)
        # - Card encodings (one-hot or embeddings)
        # - Troop positions and types (grid or set encoding)
        # - Tower health values
        # - Match time remaining

        obs = np.zeros(self.obs_dim, dtype=np.float32)

        if state is None:
            return obs

        # Basic encoding (placeholder)
        # Index 0: Normalized elixir
        obs[0] = state.elixir / 10.0

        # Index 1: Elixir rate indicator
        obs[1] = (state.elixir_rate - 1.0) / 2.0  # 0 for 1x, 0.5 for 2x, 1 for 3x

        # Index 2: Match time (normalized)
        obs[2] = state.match_time_remaining / 300.0  # Assume 5 min max

        # Index 3-6: Friendly tower health
        for i, tower in enumerate(state.friendly_towers[:3]):
            obs[3 + i] = tower.health_ratio

        # Index 6-9: Enemy tower health
        for i, tower in enumerate(state.enemy_towers[:3]):
            obs[6 + i] = tower.health_ratio

        # Index 9-13: Card playability (based on elixir)
        for i, card in enumerate(state.hand[:4]):
            obs[9 + i] = 1.0 if card.can_play(state.elixir) else 0.0

        # Remaining indices could encode:
        # - Troop positions (spatial grid)
        # - Card type embeddings
        # - Recent actions taken

        return obs

    def _calculate_reward(self, action: GameAction) -> float:
        """Calculate reward from state transition.

        Base reward is derived from tower damage differential.
        Verifiers provide additional reward shaping.

        Args:
            action: The action that was executed

        Returns:
            Total reward signal
        """
        base_reward = 0.0

        if self._prev_state is not None and self._current_state is not None:
            # Reward for dealing tower damage
            prev_enemy_health = self._prev_state.get_enemy_tower_health()
            curr_enemy_health = self._current_state.get_enemy_tower_health()
            damage_dealt = prev_enemy_health - curr_enemy_health

            # Penalty for taking tower damage
            prev_friendly_health = self._prev_state.get_friendly_tower_health()
            curr_friendly_health = self._current_state.get_friendly_tower_health()
            damage_taken = prev_friendly_health - curr_friendly_health

            base_reward = damage_dealt * 10.0 - damage_taken * 10.0

        # Apply verifiers for reward shaping
        for verifier in self.verifiers:
            shaped_reward = verifier.compute_reward(
                self._prev_state,
                self._current_state,
            )
            base_reward += shaped_reward * verifier.weight

        return base_reward

    def _is_match_over(self) -> bool:
        """Check if match has ended.

        Match ends when:
        - A king tower is destroyed (3 crown)
        - Time runs out
        - Victory/defeat screen is shown

        Returns:
            True if match is over
        """
        if self._current_state is None:
            return False

        # Check for king tower destruction
        for tower in self._current_state.friendly_towers:
            if tower.tower_type == "king" and tower.is_destroyed:
                return True

        for tower in self._current_state.enemy_towers:
            if tower.tower_type == "king" and tower.is_destroyed:
                return True

        # Check for time expiration
        if self._current_state.match_time_remaining <= 0:
            return True

        # TODO: Add victory/defeat screen detection

        return False

    def _is_truncated(self) -> bool:
        """Check if episode should be truncated.

        Returns:
            True if episode should be truncated
        """
        # Truncate if taking too long (e.g., stuck in menu)
        elapsed = time.time() - self._episode_start_time
        if elapsed > 600:  # 10 minute timeout
            return True

        return False

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the current game state.

        Args:
            mode: Render mode ("human" or "rgb_array")

        Returns:
            RGB array if mode is "rgb_array", else None
        """
        if mode == "rgb_array":
            return self._last_screenshot
        elif mode == "human":
            # TODO: Display screenshot in window
            pass

        return None

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions for current state.

        Returns:
            Boolean array where True indicates valid action
        """
        if self._current_state is None:
            # All actions valid if no state
            return np.ones(self.action_space.n, dtype=bool)

        playable_indices = [
            i for i, card in enumerate(self._current_state.hand)
            if card.can_play(self._current_state.elixir)
        ]

        mask = self.action_space_def.get_valid_actions_mask(playable_indices)
        return np.array(mask, dtype=bool)
