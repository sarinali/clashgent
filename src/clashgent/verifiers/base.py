"""Base classes for reward shaping verifiers."""

from abc import ABC, abstractmethod
from typing import Optional

from ..game.state import GameState


class Verifier(ABC):
    """Base class for reward shaping verifiers.

    Verifiers analyze state transitions and provide additional reward
    signals to shape learning. They allow domain-specific knowledge
    to guide the agent towards good gameplay.

    Each verifier has:
    - A compute_reward() method that returns shaped reward
    - A weight parameter to scale the reward contribution
    - Optional state for tracking across steps

    Verifiers are designed to be composable - multiple verifiers can
    be combined to create complex reward functions.

    Example verifiers:
    - TowerDamageVerifier: Reward dealing tower damage
    - ElixirEfficiencyVerifier: Reward efficient elixir usage
    - TroopTradeVerifier: Reward positive elixir trades
    - ElixirLeakVerifier: Punish sitting at max elixir

    Attributes:
        weight: Multiplier for this verifier's reward contribution
        name: Human-readable name for logging
    """

    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        """Initialize verifier.

        Args:
            weight: Reward scaling factor
            name: Optional custom name (defaults to class name)
        """
        self.weight = weight
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute_reward(
        self,
        prev_state: Optional[GameState],
        current_state: GameState,
    ) -> float:
        """Compute shaped reward for state transition.

        This method is called after each environment step with the
        previous and current game states. It should return a reward
        signal that encourages desired behavior.

        Args:
            prev_state: Game state before action (None on first step)
            current_state: Game state after action

        Returns:
            Shaped reward signal (can be positive or negative)

        Note:
            The returned reward is multiplied by self.weight before
            being added to the total reward.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset verifier state for new episode.

        Override this method if your verifier maintains state
        across timesteps (e.g., tracking cumulative metrics).
        """
        pass

    def get_info(self) -> dict:
        """Get additional info for logging.

        Override to provide debugging information about
        the verifier's internal state.

        Returns:
            Dictionary of info for logging
        """
        return {}


# ============================================================================
# Example Verifier Structures (Not Implemented)
# ============================================================================
# These provide the structure for common reward shaping strategies.
# Implement the compute_reward method to add your custom logic.
# ============================================================================


class TowerDamageVerifier(Verifier):
    """Rewards dealing tower damage, punishes taking damage.

    Positive reward when enemy tower health decreases.
    Negative reward when friendly tower health decreases.

    Args:
        damage_reward: Reward per unit of damage dealt
        damage_penalty: Penalty per unit of damage taken
    """

    def __init__(
        self,
        weight: float = 1.0,
        damage_reward: float = 10.0,
        damage_penalty: float = 10.0,
    ):
        super().__init__(weight=weight)
        self.damage_reward = damage_reward
        self.damage_penalty = damage_penalty

    def compute_reward(
        self,
        prev_state: Optional[GameState],
        current_state: GameState,
    ) -> float:
        """Compute tower damage reward.

        TODO: Implement by comparing tower health between states.
        """
        raise NotImplementedError("TowerDamageVerifier not implemented")


class ElixirEfficiencyVerifier(Verifier):
    """Rewards efficient elixir usage.

    Encourages the agent to spend elixir wisely rather than
    wasting it or letting it leak at 10.

    This could track:
    - Elixir spent vs value gained
    - Average elixir level (penalize staying at 10)
    - Elixir advantage over opponent
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight=weight)

    def compute_reward(
        self,
        prev_state: Optional[GameState],
        current_state: GameState,
    ) -> float:
        """Compute elixir efficiency reward.

        TODO: Implement elixir tracking logic.
        """
        raise NotImplementedError("ElixirEfficiencyVerifier not implemented")


class TroopTradeVerifier(Verifier):
    """Rewards positive elixir trades.

    A positive trade occurs when you spend less elixir than
    your opponent to defend a push or when your push deals
    more damage than its cost.

    This requires tracking:
    - Troops placed and their costs
    - Troops destroyed and their values
    - Net elixir advantage from trades
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight=weight)
        self._tracking_state: dict = {}

    def reset(self) -> None:
        """Reset trade tracking."""
        self._tracking_state = {}

    def compute_reward(
        self,
        prev_state: Optional[GameState],
        current_state: GameState,
    ) -> float:
        """Compute trade value reward.

        TODO: Implement troop trade tracking.
        """
        raise NotImplementedError("TroopTradeVerifier not implemented")


class ElixirLeakVerifier(Verifier):
    """Punishes sitting at 10 elixir (leaking).

    When elixir is at max (10), any generated elixir is wasted.
    This verifier encourages the agent to play cards before
    reaching max elixir.

    Args:
        leak_penalty: Penalty per step at max elixir
        threshold: Elixir level to start penalizing (default 10)
    """

    def __init__(
        self,
        weight: float = 1.0,
        leak_penalty: float = 0.1,
        threshold: float = 10.0,
    ):
        super().__init__(weight=weight)
        self.leak_penalty = leak_penalty
        self.threshold = threshold

    def compute_reward(
        self,
        prev_state: Optional[GameState],
        current_state: GameState,
    ) -> float:
        """Compute elixir leak penalty.

        TODO: Implement leak detection.
        """
        raise NotImplementedError("ElixirLeakVerifier not implemented")


class DefensivePlayVerifier(Verifier):
    """Rewards defensive plays when under attack.

    Encourages the agent to respond to enemy pushes by
    placing defensive troops on their side of the arena.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight=weight)

    def compute_reward(
        self,
        prev_state: Optional[GameState],
        current_state: GameState,
    ) -> float:
        """Compute defensive play reward.

        TODO: Implement by detecting enemy pushes and defensive responses.
        """
        raise NotImplementedError("DefensivePlayVerifier not implemented")


class CounterPlayVerifier(Verifier):
    """Rewards playing effective counters to enemy troops.

    Encourages the agent to learn proper counter relationships
    (e.g., skeleton army vs single-target troops).
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight=weight)

    def compute_reward(
        self,
        prev_state: Optional[GameState],
        current_state: GameState,
    ) -> float:
        """Compute counter play reward.

        TODO: Implement counter detection using game knowledge.
        """
        raise NotImplementedError("CounterPlayVerifier not implemented")
