"""Actor-Critic policy networks for PPO."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Actor-Critic policy network for PPO.

    A neural network with shared feature extraction and separate
    actor (policy) and critic (value) heads.

    Architecture:
    - Shared encoder: Processes observations into features
    - Actor head: Outputs action logits
    - Critic head: Outputs state value estimate

    Attributes:
        encoder: Shared feature extraction layers
        actor: Policy head for action selection
        critic: Value head for state evaluation
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        """Initialize Actor-Critic network.

        Args:
            obs_dim: Dimension of observation vector
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer size
            num_layers: Number of hidden layers in encoder
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Build shared encoder
        encoder_layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network.

        Args:
            obs: Observation tensor (B, obs_dim)

        Returns:
            Tuple of:
            - action_logits: (B, action_dim)
            - value: (B, 1)
        """
        features = self.encoder(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            obs: Observation tensor (B, obs_dim)
            deterministic: If True, select most likely action
            action_mask: Optional boolean mask of valid actions

        Returns:
            Tuple of:
            - action: Selected action indices (B,)
            - value: State value estimates (B, 1)
        """
        action_logits, value = self.forward(obs)

        # Apply action mask if provided
        if action_mask is not None:
            # Set invalid action logits to very negative value
            action_logits = action_logits.masked_fill(
                ~action_mask,
                float("-inf"),
            )

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            dist = Categorical(logits=action_logits)
            action = dist.sample()

        return action, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy for given actions.

        Used during PPO update to compute policy gradient.

        Args:
            obs: Observation tensor (B, obs_dim)
            actions: Action indices (B,)
            action_mask: Optional boolean mask of valid actions

        Returns:
            Tuple of:
            - log_probs: Log probabilities of actions (B,)
            - entropy: Policy entropy (B,)
            - value: State value estimates (B, 1)
        """
        action_logits, value = self.forward(obs)

        # Apply action mask if provided
        if action_mask is not None:
            action_logits = action_logits.masked_fill(
                ~action_mask,
                float("-inf"),
            )

        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate for observations.

        Args:
            obs: Observation tensor (B, obs_dim)

        Returns:
            Value estimates (B, 1)
        """
        features = self.encoder(obs)
        value = self.critic(features)
        return value


class CNNActorCritic(nn.Module):
    """Actor-Critic with CNN encoder for image observations.

    Processes raw screenshots directly instead of encoded vectors.
    Useful for end-to-end training without separate vision system.

    Attributes:
        cnn: Convolutional feature extractor
        encoder: Fully connected feature processor
        actor: Policy head
        critic: Value head
    """

    def __init__(
        self,
        input_channels: int = 3,
        action_dim: int = 1000,
        hidden_dim: int = 512,
    ):
        """Initialize CNN Actor-Critic.

        Args:
            input_channels: Number of input image channels
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # CNN feature extractor (Nature DQN style)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size (depends on input image size)
        # Assuming 84x84 input: output is 64 * 7 * 7 = 3136
        cnn_output_dim = 64 * 7 * 7

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(cnn_output_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network.

        Args:
            obs: Image tensor (B, C, H, W)

        Returns:
            Tuple of action logits and values
        """
        cnn_features = self.cnn(obs)
        features = self.encoder(cnn_features)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_logits, value = self.forward(obs)

        if action_mask is not None:
            action_logits = action_logits.masked_fill(
                ~action_mask,
                float("-inf"),
            )

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            dist = Categorical(logits=action_logits)
            action = dist.sample()

        return action, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probs and entropy for given actions."""
        action_logits, value = self.forward(obs)

        if action_mask is not None:
            action_logits = action_logits.masked_fill(
                ~action_mask,
                float("-inf"),
            )

        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, value
