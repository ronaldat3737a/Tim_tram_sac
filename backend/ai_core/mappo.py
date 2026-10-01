"""MAPPO networks (CTDE) for MultiAgentEVEnv.

One Actor shared by every app EV (parameter sharing): local observation ->
Categorical distribution over stations. One centralized Critic: global
state (MultiAgentEVEnv.state(): station loads plus time progress) -> V.
Input widths come from the env's spaces (train_mappo.build_agent), so they
always match its observation and state layout. Every method accepts either a single
row or a batch of rows and always works on 2-D tensors internally, so a
decision batch with a single agent is just a batch of size 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

DEFAULT_HIDDEN_SIZES = (64, 64)


def _mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int, output_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for size in hidden_sizes:
        layers += [_init(nn.Linear(last, size), gain=np.sqrt(2)), nn.Tanh()]
        last = size
    layers.append(_init(nn.Linear(last, output_dim), gain=output_gain))
    return nn.Sequential(*layers)


def _init(layer: nn.Linear, gain: float) -> nn.Linear:
    # Orthogonal init, as in the PPO/MAPPO reference implementations.
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    """Local observation -> logits over stations."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: Sequence[int]):
        super().__init__()
        # Small output gain: a near-uniform initial policy explores evenly.
        self.net = _mlp(obs_dim, hidden_sizes, num_actions, output_gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """Global state -> V(s)."""

    def __init__(self, state_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = _mlp(state_dim, hidden_sizes, 1, output_gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class MAPPOAgent:
    """Shared Actor + centralized Critic, with save/load."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_actions: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        device: str | torch.device = "cpu",
    ):
        # Plain ints (gymnasium's Discrete.n is a numpy int), so a saved
        # checkpoint loads with torch.load(weights_only=True).
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.device = torch.device(device)
        self.actor = Actor(self.obs_dim, self.num_actions, self.hidden_sizes).to(self.device)
        self.critic = Critic(self.state_dim, self.hidden_sizes).to(self.device)
        # Free-form training info stored alongside the weights (see save()).
        self.metadata: dict = {}

    # --- inference -----------------------------------------------------

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Actions and their log-probabilities for one observation
        (obs_dim,) or a batch (n, obs_dim). Always returns 1-D arrays of
        length n (1 for a single observation). deterministic picks the most
        likely station (argmax), as served by the API."""
        dist = self._distribution(self._as_batch(obs, self.obs_dim))
        actions = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return actions.cpu().numpy(), dist.log_prob(actions).cpu().numpy()

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> np.ndarray:
        """V for one global state (state_dim,) or a batch (n, state_dim);
        always a 1-D array."""
        return self.critic(self._as_batch(global_state, self.state_dim)).cpu().numpy()

    # --- training ------------------------------------------------------

    def evaluate_actions(
        self, obs: torch.Tensor, global_state: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(log_prob, entropy, value) for a batch of agent rows, with
        gradients. Row i pairs agent i's local obs with the global state of
        the decision batch it acted in."""
        obs = self._as_batch(obs, self.obs_dim)
        global_state = self._as_batch(global_state, self.state_dim)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).reshape(-1)
        dist = self._distribution(obs)
        return dist.log_prob(actions), dist.entropy(), self.critic(global_state)

    # --- persistence ---------------------------------------------------

    def save(self, path: str | Path, **metadata) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "state_dim": self.state_dim,
                "num_actions": self.num_actions,
                "hidden_sizes": list(self.hidden_sizes),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> MAPPOAgent:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        agent = cls(
            checkpoint["obs_dim"],
            checkpoint["state_dim"],
            checkpoint["num_actions"],
            checkpoint["hidden_sizes"],
            device=device,
        )
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.metadata = checkpoint.get("metadata", {})
        return agent

    # --- helpers -------------------------------------------------------

    def _distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def _as_batch(self, rows, width: int) -> torch.Tensor:
        """Float tensor of shape (n, width) from one row or a batch."""
        tensor = torch.as_tensor(np.asarray(rows, dtype=np.float32) if not torch.is_tensor(rows) else rows)
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2 or tensor.shape[1] != width:
            raise ValueError(f"expected shape (n, {width}) or ({width},), got {tuple(tensor.shape)}")
        return tensor
