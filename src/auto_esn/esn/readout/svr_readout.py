import torch
from torch import nn, Tensor


class SVDReadout(nn.Module):
    def __init__(self, total_hidden_size: int, output_dim: int, regularization: float = 1.):
        super().__init__()
        self.readout = nn.Linear(total_hidden_size, output_dim)
        self.regularization = regularization

    def forward(self, input: Tensor) -> Tensor:
        return self.readout(input)

    def fit(self, input: Tensor, target: Tensor):
        X = torch.ones(input.size(0), 1 + input.size(1), device=target.device)
        X[:, :-1] = input
        W = self._solve_svd(X, target, self.regularization)
        self.readout.bias = nn.Parameter(W[:, -1], requires_grad=False)
        self.readout.weight = nn.Parameter(W[:, :-1], requires_grad=False)

    def _solve_svd(self, X: Tensor, y: Tensor, alpha: float) -> Tensor:
        # implementation taken from scikit-learn
        y = y[:, 0, :]  # ignore batch
        U, s, V = torch.svd(X)
        idx = s > 1e-15  # same default value as scipy.linalg.pinv
        s_nnz = s[idx][:, None]
        UTy = U.T @ y
        d = torch.zeros(s.size(0), 1, device=X.device)
        d[idx] = s_nnz / (s_nnz ** 2 + alpha)
        d_UT_y = d * UTy

        return (V @ d_UT_y).T

    def to_cuda(self):
        self.readout.to('cuda')
