import torch
from torch import nn

class pHSIC(nn.Module):
    def __init__(self, x_kernel, y_kernel, x_processing=None, y_processing=None, mode='biased',
                 x_sigma=None, y_sigma=None):
        super().__init__()

        self.x_kernel = x_kernel
        self.y_kernel = y_kernel
        self.x_processing = x_processing
        self.y_processing = y_processing
        self.mode = mode
        self.x_sigma = x_sigma
        self.y_sigma = y_sigma

    def forward(self, x, y):

        return hsic(x, y, self.x_kernel, self.y_kernel, self.x_sigma, self.y_sigma, self.mode)

def estimate_hsic(a_matrix, b_matrix, mode='biased'):
    """
    Estimates HSIC (if mode='biased') or pHSIC (if mode='plausible') between variables A and B.
    :param a_matrix:    torch.Tensor, a_matrix_ij = k(a_i,a_j), symmetric
    :param b_matrix:    torch.Tensor, b_matrix_ij = k(b_i,b_j), symmetric, must be the same size as a_matrix
    :param mode:        str, 'biased' (HSIC) or 'plausible' (pHSIC)
    :return: float, HSIC or pHSIC estimate
    """
    if mode == 'biased':
        a_vec = a_matrix.mean(dim=0)
        b_vec = b_matrix.mean(dim=0)
        # same as tr(HAHB)/m^2 for A=a_matrix, B=b_matrix, H=I - 11^T/m (centering matrix)
        return (a_matrix * b_matrix).mean() - 2 * (a_vec * b_vec).mean() + a_vec.mean() * b_vec.mean()
    if mode == 'plausible':
        # same as tr((A - mean(A))(B - mean(B)))/m^2
        return ((a_matrix - a_matrix.mean()) * b_matrix).mean()

    raise NotImplementedError('mode must be either biased or plausible, but %s was given' % mode)


def hsic(x, y, x_kernel, y_kernel, x_sigma, y_sigma, mode='biased'):
    """
    Estimates the kernelized bottleneck objective between activations z and labels y.
    :param z:         torch.Tensor, activations, shape (batch_size, ...)
    :param y:         torch.Tensor, labels, shape (batch_size, ...)
    :param z_kernel:  Kernel, kernel to use for z
    :param y_kernel:  Kernel, kernel to use for y
    :param gamma:     float, balance parameter (float)
    :param mode:      str, 'biased' (HSIC) or 'plausible' (pHSIC)
    :return: float, HSIC(z,z) - gamma * HSIC(z, y) (or pHSIC, if mode='plausible')
    """
    x_matrix = x_kernel.compute(x, x_sigma)
    y_matrix = y_kernel.compute(y, y_sigma)
    return estimate_hsic(y_matrix, x_matrix, mode)


def compute_linear_kernel(batch):
    """
    Computes the linear kernel between input vectors.
    :param batch: torch.Tensor, input vectors
    :return: torch.Tensor, matrix A such that A_ij = batch[i]^T batch[j] (for flattened batch[i] and batch[j])
    """
    return torch.mm(batch.view(batch.shape[0], -1), batch.view(batch.shape[0], -1).transpose(0, 1))


def compute_pdist_matrix(batch, p=2.0):
    """
    Computes the matrix of pairwise distances w.r.t. p-norm
    :param batch: torch.Tensor, input vectors
    :param p:     float, norm parameter, such that ||x||p = (sum_i |x_i|^p)^(1/p)
    :return: torch.Tensor, matrix A such that A_ij = ||batch[i] - batch[j]||_p (for flattened batch[i] and batch[j])
    """
    mat = torch.zeros(batch.shape[0], batch.shape[0], device=batch.device)
    ind = torch.triu_indices(batch.shape[0], batch.shape[0], offset=1, device=batch.device)
    mat[ind[0], ind[1]] = torch.pdist(batch.view(batch.shape[0], -1), p=p)

    return mat + mat.transpose(0, 1)


class Kernel:
    """
    Base class for different kernels.
    """
    def __init__(self):
        pass

    def compute(self, batch):
        raise NotImplementedError()


class LinearKernel(Kernel):
    """
    Linear kernel: k(a_i, a_j) = a_i^T a_j
    """
    def __init__(self):
        super().__init__()

    def compute(self, batch, sigma=None):
        """
        Computes the linear kernel between input vectors.
        :param batch: torch.Tensor, input vectors
        :return: torch.Tensor, matrix A such that A_ij = batch[i]^T batch[j] (for flattened batch[i] and batch[j])
        """
        return compute_linear_kernel(batch)


class GaussianKernel(Kernel):
    """
    Gaussian kernel: k(a_i, a_j) = epx(-||a_i - a_j||^2 / (2 sigma^2))
    """
    def __init__(self, sigma=None):
        """
        :param sigma: float, Gaussian kernel sigma
        """
        super().__init__()

    def compute(self, batch, sigma=None):
        """
        Computes the Gaussian kernel between input vectors.
        :param batch: torch.Tensor, input vectors
        :return: torch.Tensor, matrix A such that A_ij = exp(-||batch[i] - batch[j]||^2 / (2 sigma^2))
            (for flattened batch[i] and batch[j])
        """
        if sigma is None:
            sigma = (torch.nn.functional.pdist(batch)).median()
        return torch.exp(-(compute_pdist_matrix(batch, p=2.0)) ** 2 / (2.0 * sigma ** 2))


class CosineSimilarityKernel(Kernel):
    """
    Cosine similarity kernel: k(a_i, a_j) = a_i^T a_j / (||a_i||_2 ||a_j||_2)
    """
    def __init__(self):
        super().__init__()
        self.eps = 1e-6  # s.t. a_i / ||a_i||_2 == 0.0 if ||a_i||_2 == 0.0

    def compute(self, batch):
        """
        Computes the cosine similarity kernel between input vectors.
        :param batch: torch.Tensor, input vectors
        :return: torch.Tensor, matrix A such that A_ij = batch[i]^T batch[j] / (||batch[i]||_2 ||batch[j]||_2)
            (for flattened batch[i] and batch[j])
        """
        normalization = torch.norm(batch.view(batch.shape[0], -1), dim=-1, keepdim=True, p=2.0)
        normalization = normalization + self.eps * (normalization <= 0.0)
        return compute_linear_kernel(batch.view(batch.shape[0], -1) / normalization)