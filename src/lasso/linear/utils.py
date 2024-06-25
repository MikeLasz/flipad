import warnings

import torch


def qr(A):
    try:
        # only for newer pytorch versions
        return torch.linalg.qr(A, mode="reduced")
    except AttributeError:
        return torch.qr(A, some=True)


def lstsq(b, A):
    m, n = A.shape[-2:]
    if m < n:
        # solve least-norm problem
        Q, R = qr(A.transpose(-1, -2))
        d = torch.triangular_solve(b, R.transpose(-1, -2), upper=False)[0]
        x = torch.matmul(Q, d)
    else:
        # solve least-squares problem
        Q, R = qr(A)
        d = torch.matmul(Q.transpose(-1, -2), b)
        x = torch.triangular_solve(d, R)[0]
    return x


def ridge(b, A, alpha=1e-4):
    # right-hand side
    rhs = torch.matmul(A.T, b)
    # regularized gram matrix
    M = torch.matmul(A.T, A)
    M.diagonal().add_(alpha)
    # solve
    L, info = torch.linalg.cholesky_ex(M)
    if info != 0:
        raise RuntimeError("The Gram matrix is not positive definite. " "Try increasing 'alpha'.")
    x = torch.cholesky_solve(rhs, L)
    return x


def batch_cholesky_solve(b, A):
    """
    Solve a batch of PSD linear systems, with a unique matrix A_k for
    each batch entry b_k
    """
    assert b.dim() == 2  # [B,D]
    assert A.dim() == 3  # [B,D,D]
    b = b.unsqueeze(2)  # [B,D,1]
    L, info = torch.linalg.cholesky_ex(A)
    if torch.all(info == 0):
        x = torch.cholesky_solve(b, L)  # [B,D,1]
    else:
        warnings.warn("Cholesky factorization failed. Reverting to LU " "decomposition...")
        x = torch.linalg.solve(A, b)  # [B,D,1]
    return x.squeeze(2)
