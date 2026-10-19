import numpy as np
import torch

from functions.svd_operators import Deblurring, SRConv
from functions.fft_operators import DeblurringFFT


def _bicubic_kernel(factor, a=-0.5):
    def cubic(x):
        x = abs(x)
        if x <= 1:
            return (a + 2) * x**3 - (a + 3) * x**2 + 1
        if x < 2:
            return a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a
        return 0
    k = np.array([cubic((1 / factor) * (ii - np.floor(factor * 4 / 2) + 0.5)) for ii in range(factor * 4)])
    return k / np.sum(k)


def get_operator(task, deg_scale, img_size, device, index=0):
    """Forward operator A for `task`.
    sr_bicubic:    bicubic super-resolution, deg_scale = scale factor
    deblur_gauss:  Gaussian blur, deg_scale = standard deviation
    deblur_motion: motion blur (intensity 0.5), deg_scale = kernel size; the kernel depends on `index`."""
    if task == 'sr_bicubic':
        kernel = torch.from_numpy(_bicubic_kernel(int(deg_scale))).float().to(device)
        return SRConv(kernel / kernel.sum(), 3, img_size, device, stride=int(deg_scale))
    if task == 'deblur_gauss':
        sig = deg_scale
        kernel = torch.Tensor([torch.exp(torch.Tensor([-0.5 * (k / sig) ** 2])) for k in range(-30, 30)]).to(device)
        return Deblurring(kernel / kernel.sum(), 3, img_size, device)
    if task == 'deblur_motion':
        try:
            from motionblur.motionblur import Kernel
        except ImportError as e:
            raise ImportError('Motion deblurring needs https://github.com/LeviBorodenko/motionblur '
                              'cloned into the repository root (see README).') from e
        np.random.seed(seed=index * 10)
        kernel = torch.from_numpy(Kernel(size=(deg_scale, deg_scale), intensity=0.5).kernelMatrix)
        return DeblurringFFT(kernel / kernel.sum(), 3, img_size, device)
    raise ValueError(f'unknown task: {task}')


def measurement_image(task, A, y, img_size):
    """Image-space view of the measurement for saving (A^T y for super-resolution)."""
    yv = A.At(y) if task == 'sr_bicubic' else y
    yv = yv.reshape(1, 3, img_size, img_size)
    return ((yv - yv.min()) / (yv.max() - yv.min() + 1e-8)).clamp(0, 1)
