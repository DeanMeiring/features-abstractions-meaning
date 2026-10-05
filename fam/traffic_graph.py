"""A small graph network that forecasts every road sensor at once (a mini DCRNN).

    each sensor's last hour + time of day --> its own hidden state
        --> 2 rounds of passing messages along road links, both directions
            (neighbours, then neighbours of neighbours)
        --> forecast of each sensor's speed, as a change from its current speed

use_graph=False gives the same network with the message passing switched off,
so each sensor only sees its own data: the fair "without neighbours" control.
Road links are weighted by distance (fam/traffic.road_weights) and stored
sparsely, since each sensor only has a handful of links.
"""

import numpy as np
import torch
from torch import nn

HISTORY = 12


def _sparse_rows(weights: np.ndarray) -> torch.Tensor:
    """Row-normalised sparse matrix: row i averages over sensor i's linked sensors."""
    sums = weights.sum(axis=1, keepdims=True)
    normed = np.divide(weights, sums, out=np.zeros_like(weights), where=sums > 0)
    rows, cols = np.nonzero(normed)
    return torch.sparse_coo_tensor(np.stack([rows, cols]), normed[rows, cols].astype(np.float32), normed.shape,
                                   check_invariants=True).coalesce()


class GraphForecaster(nn.Module):
    def __init__(self, weights: np.ndarray, use_graph: bool = True, hidden: int = 64, hops: int = 2):
        super().__init__()
        self.use_graph, self.n_sensors = use_graph, len(weights)
        self.downstream = _sparse_rows(weights)    # messages from the sensors you drive to
        self.upstream = _sparse_rows(weights.T)    # messages from the sensors that drive to you
        self.sensor = nn.Embedding(self.n_sensors, 8)  # lets each sensor learn its own habits
        self.inp = nn.Linear(HISTORY + 3 + 8, hidden)
        self.own = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(hops))
        self.down = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(hops))
        self.up = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(hops))
        self.head = nn.Linear(hidden, 1)

    def _spread(self, matrix, h):
        """Average neighbours' hidden states: (batch, sensors, hidden) -> same shape."""
        b, n, d = h.shape
        flat = h.permute(1, 0, 2).reshape(n, b * d)
        return torch.sparse.mm(matrix, flat).reshape(n, b, d).permute(1, 0, 2)

    def forward(self, x):
        """x: (batch, sensors, 12 scaled speeds + 3 time features) -> (batch, sensors) scaled change."""
        ids = self.sensor.weight.expand(len(x), -1, -1)
        h = torch.relu(self.inp(torch.cat([x, ids], dim=-1)))
        for own, down, up in zip(self.own, self.down, self.up):
            message = own(h)
            if self.use_graph:
                message = message + down(self._spread(self.downstream, h)) + up(self._spread(self.upstream, h))
            h = h + torch.relu(message)
        return self.head(h).squeeze(-1)


def batch_inputs(speeds: torch.Tensor, time_features: torch.Tensor, ends: torch.Tensor, mean: float, scale: float):
    """Inputs for forecasts made at time steps `ends`: (batch, sensors, 15)."""
    idx = ends[:, None] + torch.arange(-HISTORY + 1, 1)                 # (batch, 12)
    lags = (speeds[idx] - mean) / scale                                 # (batch, 12, sensors)
    t = time_features[ends][:, None, :].expand(-1, speeds.shape[1], -1)  # (batch, sensors, 3)
    return torch.cat([lags.permute(0, 2, 1), t], dim=-1)


def train(speeds: np.ndarray, time_features: np.ndarray, weights: np.ndarray, train_ends: np.ndarray,
          val_ends: np.ndarray, steps: int, use_graph: bool, epochs: int = 10, batch_size: int = 32,
          seed: int = 0) -> GraphForecaster:
    """Fit on the training months; keep the epoch with the lowest validation error."""
    torch.manual_seed(seed)
    s, tf = torch.tensor(speeds), torch.tensor(time_features, dtype=torch.float32)
    mean, scale = float(speeds[:val_ends.min()].mean()), float(speeds[:val_ends.min()].std())
    model = GraphForecaster(weights, use_graph)
    model.mean, model.scale = mean, scale
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state = np.inf, None
    for epoch in range(epochs):
        model.train()
        order = torch.tensor(train_ends)[torch.randperm(len(train_ends))]
        for start in range(0, len(order), batch_size):
            ends = order[start:start + batch_size]
            target, last = s[ends + steps], s[ends]
            predicted = last + model(batch_inputs(s, tf, ends, mean, scale)) * scale
            known = target > 0  # 0 = missing reading
            loss = (predicted - target).abs()[known].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = mean_error(model, speeds, time_features, val_ends, steps)
        print(f"    epoch {epoch + 1}/{epochs}: validation error {val:.3f} mph")
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def predict(model: GraphForecaster, speeds: np.ndarray, time_features: np.ndarray, ends: np.ndarray,
            steps: int, batch_size: int = 256) -> np.ndarray:
    """Forecasts (len(ends), sensors) in mph."""
    model.eval()
    s, tf = torch.tensor(speeds), torch.tensor(time_features, dtype=torch.float32)
    out = []
    for start in range(0, len(ends), batch_size):
        e = torch.tensor(ends[start:start + batch_size])
        out.append((s[e] + model(batch_inputs(s, tf, e, model.mean, model.scale)) * model.scale).numpy())
    return np.concatenate(out)


def mean_error(model, speeds, time_features, ends, steps) -> float:
    predicted, actual = predict(model, speeds, time_features, ends, steps), speeds[ends + steps]
    known = actual > 0
    return float(np.abs(predicted - actual)[known].mean())
