"""Action space definition for Clash Royale."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .state import Position


class ActionType(Enum):
    """Types of actions the agent can take."""
    WAIT = "wait"           # Do nothing this frame
    PLAY_CARD = "play_card"  # Place a card on the arena


@dataclass
class GameAction:
    """An action to take in the game.

    Attributes:
        action_type: Whether to wait or play a card
        card_index: Which card to play (0-3), only used for PLAY_CARD
        target_position: Where to place the card, only used for PLAY_CARD
    """
    action_type: ActionType
    card_index: Optional[int] = None  # 0-3 for which card to play
    target_position: Optional[Position] = None

    @classmethod
    def wait(cls) -> "GameAction":
        """Create a wait action."""
        return cls(action_type=ActionType.WAIT)

    @classmethod
    def play_card(cls, card_index: int, position: Position) -> "GameAction":
        """Create a play card action.

        Args:
            card_index: Index of card to play (0-3)
            position: Normalized position to place the card

        Returns:
            GameAction for playing a card
        """
        return cls(
            action_type=ActionType.PLAY_CARD,
            card_index=card_index,
            target_position=position,
        )

    def is_valid(self) -> bool:
        """Check if this action is properly formed."""
        if self.action_type == ActionType.WAIT:
            return True
        elif self.action_type == ActionType.PLAY_CARD:
            return (
                self.card_index is not None
                and 0 <= self.card_index <= 3
                and self.target_position is not None
            )
        return False


@dataclass
class ActionSpace:
    """Defines the valid action space for the agent.

    The action space is discretized for easier learning:
    - Action 0: WAIT (do nothing)
    - Actions 1 to N: PLAY_CARD at various positions

    For PLAY_CARD actions, the arena is divided into a grid.
    Each card can be placed at any grid position on the friendly side.

    Total actions = 1 (wait) + 4 (cards) * grid_width * placement_height

    Attributes:
        grid_width: Number of horizontal grid cells
        grid_height: Number of vertical grid cells for placement
        arena_y_min: Minimum y position for card placement (friendly side)
        arena_y_max: Maximum y position for card placement
    """
    grid_width: int = 18  # Horizontal resolution
    grid_height: int = 14  # Vertical resolution for placement area
    arena_y_min: float = 0.0  # Bottom of arena (friendly side)
    arena_y_max: float = 0.5  # Can only place on your half

    @property
    def num_positions(self) -> int:
        """Total number of placement positions."""
        return self.grid_width * self.grid_height

    @property
    def num_actions(self) -> int:
        """Total number of discrete actions.

        1 (wait) + 4 cards * num_positions
        """
        return 1 + 4 * self.num_positions

    def to_discrete_action(self, action_idx: int) -> GameAction:
        """Convert discrete action index to GameAction.

        Args:
            action_idx: Integer action index from 0 to num_actions-1

        Returns:
            GameAction representing the discrete action
        """
        if action_idx == 0:
            return GameAction.wait()

        # Decode card and position from action index
        action_idx -= 1  # Remove wait action
        card_index = action_idx // self.num_positions
        position_idx = action_idx % self.num_positions

        # Convert position index to grid coordinates
        grid_x = position_idx % self.grid_width
        grid_y = position_idx // self.grid_width

        # Convert grid coordinates to normalized position
        x = (grid_x + 0.5) / self.grid_width
        y = self.arena_y_min + (grid_y + 0.5) / self.grid_height * (self.arena_y_max - self.arena_y_min)

        return GameAction.play_card(card_index, Position(x, y))

    def from_game_action(self, action: GameAction) -> int:
        """Convert GameAction to discrete action index.

        Args:
            action: GameAction to convert

        Returns:
            Integer action index
        """
        if action.action_type == ActionType.WAIT:
            return 0

        # Encode card and position to action index
        assert action.card_index is not None
        assert action.target_position is not None

        # Convert normalized position to grid coordinates
        grid_x = int(action.target_position.x * self.grid_width)
        grid_x = max(0, min(self.grid_width - 1, grid_x))

        y_normalized = (action.target_position.y - self.arena_y_min) / (self.arena_y_max - self.arena_y_min)
        grid_y = int(y_normalized * self.grid_height)
        grid_y = max(0, min(self.grid_height - 1, grid_y))

        position_idx = grid_y * self.grid_width + grid_x
        action_idx = 1 + action.card_index * self.num_positions + position_idx

        return action_idx

    def get_valid_actions_mask(self, playable_card_indices: list[int]) -> list[bool]:
        """Get mask of valid actions given playable cards.

        Args:
            playable_card_indices: List of card indices (0-3) that can be played

        Returns:
            Boolean list where True means action is valid
        """
        mask = [False] * self.num_actions
        mask[0] = True  # Wait is always valid

        for card_idx in playable_card_indices:
            start_idx = 1 + card_idx * self.num_positions
            end_idx = start_idx + self.num_positions
            for i in range(start_idx, end_idx):
                mask[i] = True

        return mask
