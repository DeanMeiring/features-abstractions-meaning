"""A small autoencoder: squeeze a customer row into a few numbers, then rebuild it.

    customer row (many numbers) -> ENCODER -> code (few numbers) -> DECODER -> rebuilt row

Training only asks one thing: make the rebuilt row match the original. Because
the code is so small, the network has to keep what matters and drop the rest.
After training we throw the decoder away and keep the encoder: it turns any
customer into a short code, which is the "word" we store in the library.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn


def to_numbers(X: pd.DataFrame) -> pd.DataFrame:
    """Neural nets only read numbers, so turn each category into 0/1 columns.

    e.g. InternetService = "DSL" / "Fiber optic" / "No" becomes three columns
    InternetService_DSL, InternetService_Fiber optic, InternetService_No.
    """
    return pd.get_dummies(X, dtype=float)


class AutoEncoder(nn.Module):
    def __init__(self, n_inputs: int, code_size: int):
        super().__init__()
        # Encoder: many numbers -> 16 -> code_size
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, code_size),
        )
        # Decoder: the mirror image, code_size -> 16 -> many numbers
        self.decoder = nn.Sequential(
            nn.Linear(code_size, 16),
            nn.ReLU(),
            nn.Linear(16, n_inputs),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class CustomerEncoder:
    """Learns codes from training rows only, then encodes any rows."""

    def __init__(self, code_size: int = 4, epochs: int = 200, seed: int = 42):
        self.code_size = code_size
        self.epochs = epochs
        self.seed = seed

    def fit(self, X: pd.DataFrame) -> "CustomerEncoder":
        torch.manual_seed(self.seed)
        nums = to_numbers(X)
        self.columns = nums.columns

        # Put every column on a similar scale (mean 0, spread 1). Otherwise
        # TotalCharges (up to ~8000) would drown out the 0/1 columns.
        self.mean = nums.mean()
        self.std = nums.std().replace(0, 1)
        x = self._scale(nums)

        self.model = AutoEncoder(x.shape[1], self.code_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()  # average squared gap between original and rebuilt row

        for _ in range(self.epochs):
            optimizer.zero_grad()
            rebuilt = self.model(x)
            loss = loss_fn(rebuilt, x)
            loss.backward()  # work out how each weight should change to shrink the gap
            optimizer.step()  # nudge the weights that way
        self.final_loss = loss.item()
        return self

    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        # Same 0/1 columns as in training (a category missing here becomes all 0s).
        nums = to_numbers(X).reindex(columns=self.columns, fill_value=0.0)
        with torch.no_grad():
            codes = self.model.encoder(self._scale(nums)).numpy()
        names = [f"code_{i}" for i in range(self.code_size)]
        return pd.DataFrame(codes, columns=names, index=X.index)

    def _scale(self, nums: pd.DataFrame) -> torch.Tensor:
        scaled = (nums - self.mean) / self.std
        return torch.tensor(scaled.to_numpy(dtype=np.float32))
