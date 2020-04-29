import torch


def spectral_norm(tensor):
    u, s, v = torch.svd(tensor, compute_uv=False)
    norm = s[0]  # max eigenvalue is on 1 place
    return tensor / norm


def eigen_norm(tensor):
    abs_eigs = (torch.eig(tensor)[0] ** 2).sum(1).sqrt()
    return tensor / torch.max(abs_eigs)
