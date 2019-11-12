import os

import torch
import numpy as np
import scipy.linalg
from fastai import *
from fastai.core import *
from fastai.torch_core import *
from torch.autograd import Function

from .inception import InceptionV3

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

def get_activations(imgs, model, dims=2048, cuda=True):
    model.cuda()  
    
    n_imgs = len(imgs)
    pred_arr = torch.empty((n_imgs, dims))

    imgs = imgs.cuda()

    pred = model(imgs)[0]
    
    pred_arr = pred.data.view(n_imgs, -1)

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2

    covmean = sqrtm(torch.mm(sigma1, sigma2))
    
    tr_covmean = torch.trace(covmean)

    fid = (torch.dot(diff, diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean)
    return fid


def calculate_activation_statistics(img_tensor, model,
                                    dims=2048, cuda=True):
    act = get_activations(img_tensor, model)
    mu = act.mean(dim=0)
    sigma = torch.tensor(np.cov(act, rowvar=False))   
    return mu, sigma


def fid(input, target, cuda=True, dim=2048):
    """Calculates the FID of the outputs and the targets"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dim]
    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    # m vector size dims 
    # s matrix size dims x dims
    m1, s1 = calculate_activation_statistics(input, model, dim, cuda)
    m2, s2 = calculate_activation_statistics(target, model, dim, cuda)
    
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

class FIDLoss(nn.Module):
    def __init__(self):
        super(FIDLoss, self).__init__()

    def forward(self, input, target):
        return fid(input, target)