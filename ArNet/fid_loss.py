import torch
import numpy as np
import scipy.linalg
from torch.autograd import Function
import torch.nn.functional as F
from .inception import InceptionV3
# from .ssim import msssim


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).type_as(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        print('bog')
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.numpy().astype(np.float_)
            gm = grad_output.data.numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
        return grad_input


sqrtm = MatrixSquareRoot.apply


def get_activations(imgs, model, dim=2048):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- imgs        : Tensor of images
    -- model       : Instance of inception model
    -- dims        : Dimensionality of features returned by Inception

    Returns:
    -- A pytorch tensor of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    n_imgs = len(imgs)
    pred_arr = torch.empty((n_imgs, dim))

    imgs = imgs.cuda()

    pred = model(imgs)[0]

    pred_arr = pred.data.view(n_imgs, -1)

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Pytorch tensor containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    diff = mu1 - mu2

    covmean = sqrtm(torch.mm(sigma1, sigma2))

    tr_covmean = torch.trace(covmean)

    fid = (torch.dot(diff, diff) + torch.trace(sigma1) +
           torch.trace(sigma2) - 2 * tr_covmean)
    fid.requires_grad = True
    return fid


def calculate_activation_statistics(img_tensor, model,
                                    dim=2048, cuda=True):
    """Calculation of the statistics used by the FID.
    Params:
    -- img_tensor  : Pytorch tensor of images
    -- model       : Instance of inception model
    -- dims        : Dimensionality of features returned by Inception

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(img_tensor, model, dim)
    mu = act.mean(dim=0)
    sigma = torch.tensor(np.cov(act.cpu().numpy(), rowvar=False))
    return mu, sigma


def fid(input, target, dim=2048):
    """Calculates the FID of the inputs and the targets"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dim]
    model = InceptionV3([block_idx]).cuda()
    model.eval()
    model.requires_grad_ = False

    lambda1 = 1e-3
    # m vector size dims
    # s matrix size dims x dims
    m1, s1 = calculate_activation_statistics(input, model, dim)
    m2, s2 = calculate_activation_statistics(target, model, dim)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    base_loss = F.l1_loss(input, target).cuda()
    # ms = msssim(input, target).cuda()
    # print("Valore di fid " + str(lambda1 * fid_value))
    # print("Valore di mse " + str(lambda2 * base_loss))

    return (lambda1 * fid_value).float() + (base_loss).float()
