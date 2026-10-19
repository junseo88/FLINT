# SVD-based linear degradation operators.
# Adapted from DDRM (https://github.com/bahjat-kawar/ddrm, MIT License) and
# DDPG (https://github.com/tirer-lab/DDPG).
import math

import torch


class A_functions:
    """Implicit SVD A = U diag(s) V^T of a linear operator.
    Inputs have shape (B, ...); outputs have shape (B, D)."""

    def V(self, vec):
        raise NotImplementedError()

    def Vt(self, vec):
        raise NotImplementedError()

    def U(self, vec):
        raise NotImplementedError()

    def Ut(self, vec):
        raise NotImplementedError()

    def singulars(self):
        raise NotImplementedError()

    def add_zeros(self, vec):
        raise NotImplementedError()

    def A(self, vec):
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])

    def At(self, vec):
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))

    def A_pinv_add_eta(self, vec, eta=0):
        """Regularized pseudo-inverse V diag(s / (s^2 + eta)) U^T."""
        temp = self.Ut(vec)
        singulars = self.singulars()
        factors = singulars / (singulars * singulars + eta)
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))


class SRConv(A_functions):
    """Super-resolution: separable blur (e.g. bicubic) followed by stride-`stride` subsampling."""

    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim,
                            dim)).reshape(v.shape[0], self.channels, M.shape[0], dim)

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim,
                            dim), M).reshape(v.shape[0], self.channels, dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, stride=1):
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        # 1D convolution + subsampling matrix with reflective padding
        A_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride // 2, img_dim + stride // 2, stride):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                j_effective = j
                if j_effective < 0:
                    j_effective = -j_effective - 1
                if j_effective >= img_dim:
                    j_effective = (img_dim - 1) - (j_effective - img_dim)
                A_small[i // stride, j_effective] += kernel[j - i + kernel.shape[0] // 2]
        self.U_small, self.singulars_small, self.V_small = torch.svd(A_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        self._singulars = torch.matmul(self.singulars_small.reshape(small_dim, 1),
                                       self.singulars_small.reshape(1, small_dim)).reshape(small_dim**2)
        # permutation matching the singular values (P_1 in DDRM, App. D.5)
        self._perm = torch.Tensor([self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim)]
                                  + [self.img_dim * i + j for i in range(self.small_dim)
                                     for j in range(self.small_dim, self.img_dim)]).to(device).long()

    def V(self, vec):
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, :self._perm.shape[0], :]
        temp[:, self._perm.shape[0]:, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, self._perm.shape[0]:, :]
        temp = temp.permute(0, 2, 1)
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(vec.shape[0], self.channels, -1)
        temp[:, :, :self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        temp = torch.zeros(vec.shape[0], self.small_dim**2, self.channels, device=vec.device)
        temp[:, :self.small_dim**2, :] = vec.clone().reshape(vec.shape[0], self.small_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1), self.small_dim).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone(), self.small_dim)
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(vec.shape[0], self.channels, -1)
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


class Deblurring(A_functions):
    """Separable blur (e.g. Gaussian) with zero padding."""

    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                            self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                            self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, ZERO=3e-2):
        self.img_dim = img_dim
        self.channels = channels
        A_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            # floor/ceil keeps the right-most tap of odd-sized kernels
            for j in range(i - math.floor(kernel.shape[0] / 2), i + math.ceil(kernel.shape[0] / 2)):
                if j < 0 or j >= img_dim:
                    continue
                A_small[i, j] = kernel[j - i + kernel.shape[0] // 2]
        self.U_small, self.singulars_small, self.V_small = torch.svd(A_small, some=False)
        self.singulars_small[self.singulars_small < ZERO] = 0
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1),
                                       self.singulars_small.reshape(1, img_dim)).reshape(img_dim**2)
        self._singulars, self._perm = self._singulars.sort(descending=True)

    def V(self, vec):
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)
