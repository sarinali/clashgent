"""Registry for verifier plugins."""

from typing import Type, Optional

from .base import Verifier


class VerifierRegistry:
    """Registry for discovering and creating verifiers.

    Provides a centralized way to:
    - Register verifier classes by name
    - Discover available verifiers
    - Create verifier instances by name

    Usage:
        # Register a verifier
        @VerifierRegistry.register("tower_damage")
        class TowerDamageVerifier(Verifier):
            ...

        # Or register manually
        VerifierRegistry.register_class("my_verifier", MyVerifier)

        # Create instances
        verifier = VerifierRegistry.create("tower_damage", weight=1.5)

        # List available verifiers
        print(VerifierRegistry.list_all())
    """

    _verifiers: dict[str, Type[Verifier]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a verifier class.

        Args:
            name: Unique name for the verifier

        Returns:
            Decorator function

        Example:
            @VerifierRegistry.register("my_verifier")
            class MyVerifier(Verifier):
                def compute_reward(self, prev, curr):
                    return 0.0
        """
        def decorator(verifier_cls: Type[Verifier]) -> Type[Verifier]:
            if not issubclass(verifier_cls, Verifier):
                raise TypeError(
                    f"Registered class must be a Verifier subclass, "
                    f"got {verifier_cls}"
                )
            cls._verifiers[name] = verifier_cls
            return verifier_cls
        return decorator

    @classmethod
    def register_class(cls, name: str, verifier_cls: Type[Verifier]) -> None:
        """Register a verifier class manually.

        Args:
            name: Unique name for the verifier
            verifier_cls: Verifier class to register
        """
        if not issubclass(verifier_cls, Verifier):
            raise TypeError(
                f"Registered class must be a Verifier subclass, "
                f"got {verifier_cls}"
            )
        cls._verifiers[name] = verifier_cls

    @classmethod
    def get(cls, name: str) -> Type[Verifier]:
        """Get verifier class by name.

        Args:
            name: Registered verifier name

        Returns:
            Verifier class

        Raises:
            KeyError: If verifier not found
        """
        if name not in cls._verifiers:
            available = ", ".join(cls._verifiers.keys()) or "none"
            raise KeyError(
                f"Unknown verifier: '{name}'. "
                f"Available verifiers: {available}"
            )
        return cls._verifiers[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> Verifier:
        """Create a verifier instance by name.

        Args:
            name: Registered verifier name
            **kwargs: Arguments to pass to verifier constructor

        Returns:
            Verifier instance

        Raises:
            KeyError: If verifier not found
        """
        verifier_cls = cls.get(name)
        return verifier_cls(**kwargs)

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered verifier names.

        Returns:
            List of verifier names
        """
        return list(cls._verifiers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered verifiers.

        Useful for testing.
        """
        cls._verifiers.clear()

    @classmethod
    def get_info(cls, name: str) -> dict:
        """Get information about a registered verifier.

        Args:
            name: Verifier name

        Returns:
            Dictionary with verifier info
        """
        verifier_cls = cls.get(name)
        return {
            "name": name,
            "class": verifier_cls.__name__,
            "module": verifier_cls.__module__,
            "docstring": verifier_cls.__doc__ or "No documentation",
        }


def create_verifier_stack(
    config: list[dict],
) -> list[Verifier]:
    """Create a stack of verifiers from configuration.

    Args:
        config: List of verifier configurations, each with:
            - name: Registered verifier name
            - weight: Optional weight (default 1.0)
            - **kwargs: Additional arguments for the verifier

    Returns:
        List of configured Verifier instances

    Example:
        config = [
            {"name": "tower_damage", "weight": 1.0},
            {"name": "elixir_leak", "weight": 0.5, "threshold": 9.5},
        ]
        verifiers = create_verifier_stack(config)
    """
    verifiers = []

    for cfg in config:
        name = cfg.pop("name")
        verifier = VerifierRegistry.create(name, **cfg)
        verifiers.append(verifier)

    return verifiers
