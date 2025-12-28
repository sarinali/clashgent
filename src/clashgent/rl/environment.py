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

    # Constants for state encoding
    MAX_TROOPS_PER_SIDE = 20
    TROOP_DIMS = 5  # x, y, type_id, health, is_air
    NUM_TROOP_CLASSES = 12  # POC: 10 troops + 2 buildings (towers detected via pixels)

    def _encode_state(self, state: GameState) -> np.ndarray:
        """Encode GameState to observation vector using fixed-slot encoding.

        Layout (256 dimensions):
        - [0-2]     Game info: elixir, elixir_rate, match_time (3)
        - [3-8]     Tower health: 3 friendly + 3 enemy (6)
        - [9-44]    Cards in hand: 4 cards × 9 dims (36)
        - [45-144]  Friendly troops: 20 slots × 5 dims (100)
        - [145-244] Enemy troops: 20 slots × 5 dims (100)
        - [245-255] Reserved (11)

        Troop slot encoding (5 dims per slot):
        - x position (0-1)
        - y position (0-1)
        - type_id (normalized by class count)
        - health_ratio (0-1)
        - is_air (0 = ground, 1 = air)

        Args:
            state: Current game state

        Returns:
            Normalized observation vector
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        if state is None:
            return obs

        # === Game info [0-2] ===
        obs[0] = state.elixir / 10.0
        obs[1] = (state.elixir_rate - 1.0) / 2.0  # 0 for 1x, 0.5 for 2x, 1 for 3x
        obs[2] = state.match_time_remaining / 300.0  # 5 min max

        # === Tower health [3-8] ===
        for i, tower in enumerate(state.friendly_towers[:3]):
            obs[3 + i] = tower.health_ratio
        for i, tower in enumerate(state.enemy_towers[:3]):
            obs[6 + i] = tower.health_ratio

        # === Cards in hand [9-44] ===
        # Each card: 8 dims for type embedding + 1 dim for elixir cost
        for i, card in enumerate(state.hand[:4]):
            base_idx = 9 + i * 9
            # Type embedding (simplified: use type index normalized)
            type_idx = self._card_type_to_index(card.card_type)
            obs[base_idx] = type_idx / self.NUM_TROOP_CLASSES
            # One-hot-ish encoding in remaining 7 dims
            if type_idx < 7:
                obs[base_idx + 1 + type_idx] = 1.0
            # Elixir cost (normalized, max 10)
            obs[base_idx + 8] = card.elixir_cost / 10.0

        # === Friendly troops [45-144] ===
        friendly_troops = self._sort_troops_by_priority(
            state.friendly_troops, state.friendly_towers
        )
        for i, troop in enumerate(friendly_troops[:self.MAX_TROOPS_PER_SIDE]):
            base_idx = 45 + i * self.TROOP_DIMS
            obs[base_idx] = troop.position.x
            obs[base_idx + 1] = troop.position.y
            obs[base_idx + 2] = self._card_type_to_index(troop.troop_type) / self.NUM_TROOP_CLASSES
            obs[base_idx + 3] = troop.health_ratio
            obs[base_idx + 4] = 1.0 if self._is_air_troop(troop.troop_type) else 0.0

        # === Enemy troops [145-244] ===
        enemy_troops = self._sort_troops_by_priority(
            state.enemy_troops, state.friendly_towers
        )
        for i, troop in enumerate(enemy_troops[:self.MAX_TROOPS_PER_SIDE]):
            base_idx = 145 + i * self.TROOP_DIMS
            obs[base_idx] = troop.position.x
            obs[base_idx + 1] = troop.position.y
            obs[base_idx + 2] = self._card_type_to_index(troop.troop_type) / self.NUM_TROOP_CLASSES
            obs[base_idx + 3] = troop.health_ratio
            obs[base_idx + 4] = 1.0 if self._is_air_troop(troop.troop_type) else 0.0

        # [245-255] Reserved for future use

        return obs

    def _card_type_to_index(self, card_type) -> int:
        """Map CardType enum to integer index for encoding."""
        from ..game.state import CardType

        type_map = {
            CardType.UNKNOWN: 0,
            CardType.KNIGHT: 1,
            CardType.ARCHERS: 2,
            CardType.SKELETONS: 3,
            CardType.GIANT: 4,
            CardType.HOG_RIDER: 5,
            CardType.VALKYRIE: 6,
            CardType.MUSKETEER: 7,
            CardType.WIZARD: 8,
            CardType.MINIONS: 9,
            CardType.GOBLIN: 10,
            CardType.CANNON: 11,
            CardType.TESLA: 12,
        }
        return type_map.get(card_type, 0)

    def _is_air_troop(self, card_type) -> bool:
        """Check if troop type is an air unit."""
        from ..game.state import CardType

        air_troops = {
            CardType.MINIONS,
            # Add more as you expand: BALLOON, BABY_DRAGON, MEGA_MINION, etc.
        }
        return card_type in air_troops

    def _sort_troops_by_priority(
        self, troops: list, friendly_towers: list
    ) -> list:
        """Sort troops by threat priority for slot allocation.

        Priority: closest to friendly towers first, then by health.
        """
        if not troops or not friendly_towers:
            return troops

        def tower_distance(troop):
            # Find minimum distance to any friendly tower
            min_dist = float('inf')
            for tower in friendly_towers:
                # Approximate tower positions (king=0.5, princesses=0.25/0.75)
                if tower.tower_type == "king":
                    tx, ty = 0.5, 0.1
                else:
                    tx, ty = 0.25, 0.2  # Simplified
                dist = ((troop.position.x - tx) ** 2 + (troop.position.y - ty) ** 2) ** 0.5
                min_dist = min(min_dist, dist)
            return min_dist

        return sorted(troops, key=lambda t: (tower_distance(t), -t.health_ratio))

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
