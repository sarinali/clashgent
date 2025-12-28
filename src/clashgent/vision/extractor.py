"""State extraction orchestration from screenshots."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..game.state import Card, GameState, Tower, Troop, TroopSide


class ObjectClassifier(ABC):
    """Abstract interface for classifying game objects in screenshots.

    Implementations should provide methods to detect and classify:
    - Troops on the battlefield
    - Cards in the player's hand
    - Elixir bar value
    - Tower health

    This interface allows swapping different vision backends:
    - Neural network based (YOLO, etc.)
    - Template matching based
    - Hybrid approaches
    """

    @abstractmethod
    def detect_troops(self, image: np.ndarray) -> list[Troop]:
        """Detect and classify all troops on the battlefield.

        Args:
            image: RGB screenshot array (H, W, 3)

        Returns:
            List of detected Troop objects with positions and types
        """
        raise NotImplementedError

    @abstractmethod
    def detect_cards(self, image: np.ndarray) -> tuple[list[Card], Optional[Card]]:
        """Detect cards in hand and the next upcoming card.

        Args:
            image: RGB screenshot array (H, W, 3)

        Returns:
            Tuple of (hand_cards, next_card) where:
            - hand_cards: List of 4 cards currently in hand
            - next_card: The next card that will be drawn
        """
        raise NotImplementedError

    @abstractmethod
    def detect_elixir(self, image: np.ndarray) -> float:
        """Detect current elixir amount from elixir bar.

        Args:
            image: RGB screenshot array (H, W, 3)

        Returns:
            Elixir value from 0.0 to 10.0
        """
        raise NotImplementedError

    @abstractmethod
    def detect_towers(self, image: np.ndarray) -> tuple[list[Tower], list[Tower]]:
        """Detect tower health for both sides.

        Args:
            image: RGB screenshot array (H, W, 3)

        Returns:
            Tuple of (friendly_towers, enemy_towers) where each list
            contains Tower objects for king and princess towers
        """
        raise NotImplementedError

    def detect_match_time(self, image: np.ndarray) -> float:
        """Detect remaining match time from UI.

        Args:
            image: RGB screenshot array (H, W, 3)

        Returns:
            Remaining time in seconds

        Note:
            Default implementation returns placeholder.
            Override for actual OCR-based time detection.
        """
        return 180.0  # Placeholder

    def detect_elixir_rate(self, image: np.ndarray) -> float:
        """Detect current elixir generation rate.

        Args:
            image: RGB screenshot array (H, W, 3)

        Returns:
            Elixir rate multiplier (1.0, 2.0, or 3.0)

        Note:
            Default implementation returns normal rate.
            Override to detect overtime/triple elixir.
        """
        return 1.0  # Placeholder


class StateExtractor:
    """Orchestrates extraction of complete GameState from screenshot.

    Combines multiple detection components to build a full game state
    representation from a single screenshot.

    Attributes:
        classifier: ObjectClassifier implementation for detection
    """

    def __init__(self, classifier: ObjectClassifier):
        """Initialize state extractor.

        Args:
            classifier: ObjectClassifier to use for detection
        """
        self.classifier = classifier

    def extract(self, screenshot: np.ndarray) -> GameState:
        """Extract complete game state from screenshot.

        Args:
            screenshot: RGB image array (H, W, 3)

        Returns:
            GameState with all detected information
        """
        # Detect all components
        troops = self.classifier.detect_troops(screenshot)
        hand, next_card = self.classifier.detect_cards(screenshot)
        elixir = self.classifier.detect_elixir(screenshot)
        friendly_towers, enemy_towers = self.classifier.detect_towers(screenshot)
        match_time = self.classifier.detect_match_time(screenshot)
        elixir_rate = self.classifier.detect_elixir_rate(screenshot)

        # Split troops by side
        friendly_troops = [t for t in troops if t.side == TroopSide.FRIENDLY]
        enemy_troops = [t for t in troops if t.side == TroopSide.ENEMY]

        return GameState(
            elixir=elixir,
            elixir_rate=elixir_rate,
            match_time_remaining=match_time,
            hand=hand,
            next_card=next_card,
            friendly_troops=friendly_troops,
            enemy_troops=enemy_troops,
            friendly_towers=friendly_towers,
            enemy_towers=enemy_towers,
            raw_screenshot=screenshot.tobytes() if screenshot is not None else None,
        )

    def extract_partial(
        self,
        screenshot: np.ndarray,
        detect_troops: bool = True,
        detect_cards: bool = True,
        detect_elixir: bool = True,
        detect_towers: bool = True,
    ) -> GameState:
        """Extract partial game state for efficiency.

        Allows skipping certain detections when not needed,
        such as during fast inference where only some information
        is required.

        Args:
            screenshot: RGB image array (H, W, 3)
            detect_troops: Whether to detect troops
            detect_cards: Whether to detect cards
            detect_elixir: Whether to detect elixir
            detect_towers: Whether to detect towers

        Returns:
            GameState with requested information
        """
        troops = []
        hand = []
        next_card = None
        elixir = 5.0  # Default
        friendly_towers = []
        enemy_towers = []

        if detect_troops:
            troops = self.classifier.detect_troops(screenshot)

        if detect_cards:
            hand, next_card = self.classifier.detect_cards(screenshot)

        if detect_elixir:
            elixir = self.classifier.detect_elixir(screenshot)

        if detect_towers:
            friendly_towers, enemy_towers = self.classifier.detect_towers(screenshot)

        friendly_troops = [t for t in troops if t.side == TroopSide.FRIENDLY]
        enemy_troops = [t for t in troops if t.side == TroopSide.ENEMY]

        return GameState(
            elixir=elixir,
            elixir_rate=1.0,
            match_time_remaining=180.0,
            hand=hand,
            next_card=next_card,
            friendly_troops=friendly_troops,
            enemy_troops=enemy_troops,
            friendly_towers=friendly_towers,
            enemy_towers=enemy_towers,
        )


class MockObjectClassifier(ObjectClassifier):
    """Mock classifier for testing without trained models.

    Returns empty or placeholder detections. Useful for:
    - Testing pipeline without vision models
    - Development and debugging
    - Unit tests
    """

    def detect_troops(self, image: np.ndarray) -> list[Troop]:
        """Return empty troop list."""
        return []

    def detect_cards(self, image: np.ndarray) -> tuple[list[Card], Optional[Card]]:
        """Return placeholder cards."""
        from ..game.state import CardType, Card

        # Return 4 unknown cards
        hand = [Card(CardType.UNKNOWN, elixir_cost=4) for _ in range(4)]
        next_card = Card(CardType.UNKNOWN, elixir_cost=4)
        return hand, next_card

    def detect_elixir(self, image: np.ndarray) -> float:
        """Return default elixir value."""
        return 5.0

    def detect_towers(self, image: np.ndarray) -> tuple[list[Tower], list[Tower]]:
        """Return full health towers."""
        friendly = [
            Tower(TroopSide.FRIENDLY, 1.0, False, "king"),
            Tower(TroopSide.FRIENDLY, 1.0, False, "princess"),
            Tower(TroopSide.FRIENDLY, 1.0, False, "princess"),
        ]
        enemy = [
            Tower(TroopSide.ENEMY, 1.0, False, "king"),
            Tower(TroopSide.ENEMY, 1.0, False, "princess"),
            Tower(TroopSide.ENEMY, 1.0, False, "princess"),
        ]
        return friendly, enemy
