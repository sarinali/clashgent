"""Game state representation for Clash Royale."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CardType(Enum):
    """Card/troop types in Clash Royale.

    POC set: 10 troops + 2 buildings. Extend as needed.
    Must match indices in ClashEnv._card_type_to_index().
    """
    UNKNOWN = "unknown"

    # POC Troops (index 1-10)
    KNIGHT = "knight"
    ARCHERS = "archers"
    SKELETONS = "skeletons"
    GIANT = "giant"
    HOG_RIDER = "hog_rider"
    VALKYRIE = "valkyrie"
    MUSKETEER = "musketeer"
    WIZARD = "wizard"
    MINIONS = "minions"
    GOBLIN = "goblin"

    # POC Buildings (index 11-12)
    CANNON = "cannon"
    TESLA = "tesla"

    # Future expansion - uncomment as needed:
    # BALLOON = "balloon"
    # WITCH = "witch"
    # SKELETON_ARMY = "skeleton_army"
    # PRINCE = "prince"
    # BABY_DRAGON = "baby_dragon"
    # PEKKA = "pekka"
    # GOLEM = "golem"
    # FIREBALL = "fireball"
    # ARROWS = "arrows"
    # ZAP = "zap"
    # INFERNO_TOWER = "inferno_tower"


class TroopSide(Enum):
    """Which side a troop belongs to."""
    FRIENDLY = "friendly"
    ENEMY = "enemy"


@dataclass
class Position:
    """Normalized position on the arena.

    Coordinates are normalized to [0.0, 1.0] where:
    - x: 0.0 is left edge, 1.0 is right edge
    - y: 0.0 is bottom (friendly side), 1.0 is top (enemy side)
    """
    x: float  # 0.0 to 1.0 normalized
    y: float  # 0.0 to 1.0 normalized

    def to_screen_coords(self, screen_width: int, screen_height: int) -> tuple[int, int]:
        """Convert normalized position to screen pixel coordinates.

        Args:
            screen_width: Width of the screen in pixels
            screen_height: Height of the screen in pixels

        Returns:
            Tuple of (x, y) pixel coordinates
        """
        return (
            int(self.x * screen_width),
            int(self.y * screen_height),
        )

    @classmethod
    def from_screen_coords(
        cls,
        x: int,
        y: int,
        screen_width: int,
        screen_height: int
    ) -> "Position":
        """Create Position from screen pixel coordinates.

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            screen_width: Width of the screen in pixels
            screen_height: Height of the screen in pixels

        Returns:
            Normalized Position
        """
        return cls(
            x=x / screen_width,
            y=y / screen_height,
        )


@dataclass
class Troop:
    """A troop/unit on the battlefield.

    Attributes:
        troop_type: The type of troop (knight, archer, etc.)
        position: Normalized position on the arena
        side: Whether this is a friendly or enemy troop
        health_ratio: Health as a ratio from 0.0 (dead) to 1.0 (full health)
    """
    troop_type: CardType
    position: Position
    side: TroopSide
    health_ratio: float = 1.0  # 0.0 to 1.0

    def is_alive(self) -> bool:
        """Check if troop is still alive."""
        return self.health_ratio > 0.0


@dataclass
class Card:
    """A card in the player's hand.

    Attributes:
        card_type: The type of card
        elixir_cost: Elixir cost to play this card
    """
    card_type: CardType
    elixir_cost: int

    def can_play(self, available_elixir: float) -> bool:
        """Check if this card can be played with available elixir."""
        return available_elixir >= self.elixir_cost


@dataclass
class Tower:
    """A tower on the arena (king tower or princess tower).

    Attributes:
        side: Whether this is a friendly or enemy tower
        health_ratio: Health as a ratio from 0.0 to 1.0
        is_destroyed: Whether the tower has been destroyed
        tower_type: Type of tower ("king" or "princess")
    """
    side: TroopSide
    health_ratio: float  # 0.0 to 1.0
    is_destroyed: bool = False
    tower_type: str = "princess"  # "king" or "princess"


@dataclass
class GameState:
    """Complete game state extracted from a screenshot.

    This dataclass represents everything the agent needs to know about
    the current game state to make a decision.

    Attributes:
        elixir: Current elixir amount (0.0 to 10.0)
        elixir_rate: Current elixir generation rate (1.0 normal, 2.0 double, 3.0 triple)
        match_time_remaining: Seconds remaining in the match
        hand: List of 4 cards currently in hand
        next_card: The next card that will enter the hand
        friendly_troops: List of friendly troops on the field
        enemy_troops: List of enemy troops on the field
        friendly_towers: List of friendly towers (1 king + 2 princess)
        enemy_towers: List of enemy towers (1 king + 2 princess)
        raw_screenshot: Optional raw screenshot bytes for debugging
    """
    elixir: float = 0.0  # 0.0 to 10.0
    elixir_rate: float = 1.0  # 1x, 2x, or 3x
    match_time_remaining: float = 180.0  # seconds

    hand: list[Card] = field(default_factory=list)  # 4 cards in hand
    next_card: Optional[Card] = None  # upcoming card

    friendly_troops: list[Troop] = field(default_factory=list)
    enemy_troops: list[Troop] = field(default_factory=list)

    friendly_towers: list[Tower] = field(default_factory=list)  # king + 2 princess
    enemy_towers: list[Tower] = field(default_factory=list)

    # Raw data for debugging
    raw_screenshot: Optional[bytes] = None

    def get_playable_cards(self) -> list[tuple[int, Card]]:
        """Get indices and cards that can be played with current elixir.

        Returns:
            List of (index, card) tuples for playable cards
        """
        return [
            (i, card) for i, card in enumerate(self.hand)
            if card.can_play(self.elixir)
        ]

    def get_friendly_tower_health(self) -> float:
        """Get total health ratio of friendly towers."""
        if not self.friendly_towers:
            return 1.0
        return sum(t.health_ratio for t in self.friendly_towers) / len(self.friendly_towers)

    def get_enemy_tower_health(self) -> float:
        """Get total health ratio of enemy towers."""
        if not self.enemy_towers:
            return 1.0
        return sum(t.health_ratio for t in self.enemy_towers) / len(self.enemy_towers)

    def get_crown_count(self) -> tuple[int, int]:
        """Get current crown count (friendly, enemy).

        Returns:
            Tuple of (friendly_crowns, enemy_crowns)
        """
        friendly_crowns = sum(1 for t in self.enemy_towers if t.is_destroyed)
        enemy_crowns = sum(1 for t in self.friendly_towers if t.is_destroyed)
        return friendly_crowns, enemy_crowns

    def is_overtime(self) -> bool:
        """Check if match is in overtime (double/triple elixir)."""
        return self.elixir_rate > 1.0
