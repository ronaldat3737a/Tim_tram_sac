"""MAPPO-GNN networks (CTDE) for GraphMultiAgentEVEnv.

Graph counterpart of backend/ai_core/mappo.py (MAPPO-MLP, unchanged), built so
that no weight's shape depends on the number of stations N:

- Actor: a GAT encoder over the EV's local station graph (each node's features
  plus the EV's own features), then one small MLP shared by every node scores
  it. The logits are those N scores, so the same weights pick among 5, 10 or
  100 stations.
- Critic: a GAT encoder over the global station load graph, mean- and
  max-pooled over the nodes (a fixed-size summary for any N) and joined with
  the global features, then an MLP -> V.

One Actor is shared by every app EV (parameter sharing). Agents of a decision
batch always choose one at a time, each seeing the earlier picks of the batch
(GraphMultiAgentEVEnv.decide_sequentially_graph), as the fixed MAPPO-MLP does
on the large scenario.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from backend.ai_core.graph_env import EDGE_DIM, EGO_DIM, GLOBAL_DIM, GLOBAL_NODE_DIM, NODE_DIM

if TYPE_CHECKING:
    from backend.ai_core.graph_env import GraphMultiAgentEVEnv

DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_LAYERS = 2
DEFAULT_HEADS = 4

Graphs = Data | Batch | Sequence[Data]


def _init(layer: nn.Linear, gain: float) -> nn.Linear:
    # Orthogonal init, as in mappo.py and the PPO/MAPPO reference implementations.
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(input_dim: int, hidden_size: int, output_dim: int, output_gain: float) -> nn.Sequential:
    return nn.Sequential(
        _init(nn.Linear(input_dim, hidden_size), gain=np.sqrt(2)),
        nn.ReLU(),
        _init(nn.Linear(hidden_size, output_dim), gain=output_gain),
    )


class GraphEncoder(nn.Module):
    """Node features -> node embeddings: a linear input layer, then GAT
    layers (edge features enter the attention) with residual connections
    and LayerNorm. Every weight is shaped by feature sizes only."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, heads: int, edge_dim: int):
        super().__init__()
        if hidden_size % heads:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by heads {heads}")
        self.input = _init(nn.Linear(input_dim, hidden_size), gain=np.sqrt(2))
        self.convs = nn.ModuleList(
            GATConv(hidden_size, hidden_size // heads, heads=heads, edge_dim=edge_dim) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input(x))
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + torch.relu(conv(h, edge_index, edge_attr)))
        return h


class GNNActor(nn.Module):
    """Local station graph + EV features -> one logit per station."""

    def __init__(self, hidden_size: int, num_layers: int, heads: int):
        super().__init__()
        self.encoder = GraphEncoder(NODE_DIM + EGO_DIM, hidden_size, num_layers, heads, EDGE_DIM)
        # Small output gain: a near-uniform initial policy explores evenly.
        self.score = _mlp(hidden_size + EGO_DIM, hidden_size, 1, output_gain=0.01)

    def forward(self, batch: Batch) -> torch.Tensor:
        """(B, N_max) logits; padding (graphs with fewer stations than the
        largest in the batch) gets the lowest float, i.e. probability 0."""
        ego = batch.u[batch.batch]
        h = self.encoder(torch.cat([batch.x, ego], dim=-1), batch.edge_index, batch.edge_attr)
        scores = self.score(torch.cat([h, ego], dim=-1)).squeeze(-1)
        logits, mask = to_dense_batch(scores, batch.batch)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class GNNCritic(nn.Module):
    """Global station load graph -> V(s)."""

    def __init__(self, hidden_size: int, num_layers: int, heads: int):
        super().__init__()
        self.encoder = GraphEncoder(GLOBAL_NODE_DIM, hidden_size, num_layers, heads, EDGE_DIM)
        self.value = _mlp(2 * hidden_size + GLOBAL_DIM, hidden_size, 1, output_gain=1.0)

    def forward(self, batch: Batch) -> torch.Tensor:
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        pooled = torch.cat(
            [global_mean_pool(h, batch.batch), global_max_pool(h, batch.batch), batch.u], dim=-1
        )
        return self.value(pooled).squeeze(-1)


class MAPPOGNNAgent:
    """Shared GNN Actor + centralized GNN Critic, with save/load. Methods
    take one graph, a list of graphs or a ready Batch."""

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        heads: int = DEFAULT_HEADS,
        device: str | torch.device = "cpu",
    ):
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.heads = int(heads)
        self.device = torch.device(device)
        self.actor = GNNActor(self.hidden_size, self.num_layers, self.heads).to(self.device)
        self.critic = GNNCritic(self.hidden_size, self.num_layers, self.heads).to(self.device)
        # Free-form training info stored alongside the weights (see save()).
        self.metadata: dict = {}

    # --- inference -----------------------------------------------------

    @torch.no_grad()
    def get_action(self, graphs: Graphs, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Actions (station indices) and their log-probabilities, one per
        graph; deterministic picks the most likely station (argmax)."""
        dist = Categorical(logits=self.actor(self._batch(graphs)))
        actions = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return actions.cpu().numpy(), dist.log_prob(actions).cpu().numpy()

    @torch.no_grad()
    def get_value(self, global_graphs: Graphs) -> np.ndarray:
        return self.critic(self._batch(global_graphs)).cpu().numpy()

    def act(self, env: GraphMultiAgentEVEnv, deterministic: bool = False) -> tuple[list[int], list[Data], np.ndarray, np.ndarray]:
        """Actions for every agent of env's current batch, decided one at a
        time (each sees the earlier picks of its batch). Returns (agents,
        the graph each chose from, actions, log_probs), in env.agents
        order."""
        agents = list(env.agents)
        log_probs: list[float] = []

        def choose(graph: Data) -> int:
            action, log_prob = self.get_action(graph, deterministic=deterministic)
            log_probs.append(float(log_prob[0]))
            return int(action[0])

        chosen, graphs = env.decide_sequentially_graph(choose)
        actions = np.array([chosen[agent] for agent in agents], dtype=np.int64)
        return agents, graphs, actions, np.array(log_probs, dtype=np.float32)

    # --- training ------------------------------------------------------

    def evaluate_actions(
        self, graphs: Graphs, global_graphs: Graphs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(log_prob, entropy, value) with gradients; row i pairs agent i's
        local graph with the global graph of the batch it acted in."""
        dist = Categorical(logits=self.actor(self._batch(graphs)))
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).reshape(-1)
        return dist.log_prob(actions), dist.entropy(), self.critic(self._batch(global_graphs))

    # --- persistence ---------------------------------------------------

    def save(self, path: str | Path, **metadata) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "heads": self.heads,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> MAPPOGNNAgent:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        agent = cls(checkpoint["hidden_size"], checkpoint["num_layers"], checkpoint["heads"], device=device)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.metadata = checkpoint.get("metadata", {})
        return agent

    def num_parameters(self) -> int:
        return sum(p.numel() for p in [*self.actor.parameters(), *self.critic.parameters()])

    # --- helpers -------------------------------------------------------

    def _batch(self, graphs: Graphs) -> Batch:
        if isinstance(graphs, Batch):
            batch = graphs
        elif isinstance(graphs, Data):
            batch = Batch.from_data_list([graphs])
        else:
            batch = Batch.from_data_list(list(graphs))
        return batch.to(self.device)
