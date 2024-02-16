"""
Computes 2-Wasserstein distance of two 2d Gaussian distributions using mean and covariance.
The 2d Gaussian is a 2nd-order approximation to the distributions due to computational constraints and curse of dimensionality.
"""
import torch

def sqrt_mat(M):
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    D_half = torch.diag(torch.sqrt(eigenvalues))
    Mhalf = eigenvectors @ D_half @ eigenvectors.T
    return Mhalf

def compute_w2(mu1, mu2, sig1, sig2):
    sig1_half = sqrt_mat(sig1)
    M = sig1_half @ sig2 @ sig1_half
    sqrtM = sqrt_mat(M)
    w2 = torch.sqrt(torch.square(mu1 - mu2).sum() + torch.trace(sig1 + sig2 - 2 * sqrtM))
    return w2
