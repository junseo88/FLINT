# FFT-based (circular) blur operator, used for motion deblurring.
import torch
import torch.fft


def _pad_and_center_kernel(kernel, H, W, device):
    mb, nb = kernel.shape[:2]
    bigK = torch.zeros(H, W, device=device, dtype=kernel.dtype)
    bigK[:mb, :nb] = kernel.to(device)
    bigK = torch.roll(bigK, (-int((mb - 1) / 2), -int((mb - 1) / 2)), dims=(0, 1))
    return bigK


class DeblurringFFT:
    def __init__(self, kernel, channels, img_dim, device, eta_reg=1e-4):
        self.img_dim = img_dim
        self.channels = channels
        self.eta_reg = eta_reg
        if kernel.dim() == 1:
            kernel = torch.matmul(kernel[:, None], kernel[None, :]) / torch.sum(kernel) ** 2
        self.kernel = kernel
        bigK = _pad_and_center_kernel(kernel, img_dim, img_dim, device)
        self._fft2K = torch.fft.fft2(bigK)
        self._fft2K_conj = torch.conj(self._fft2K)
        self._fft2K_norm2 = torch.abs(self._fft2K) ** 2

    def A(self, vec):
        I = vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        out = torch.real(torch.fft.ifft2(torch.fft.fft2(I) * self._fft2K))
        return out.reshape(vec.shape[0], -1)

    def At(self, vec):
        I = vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        out = torch.real(torch.fft.ifft2(torch.fft.fft2(I) * self._fft2K_conj))
        return out.reshape(vec.shape[0], -1)

    def A_pinv_add_eta(self, vec, eta_reg=None):
        """Regularized pseudo-inverse K^* / (|K|^2 + eta)."""
        if eta_reg is None:
            eta_reg = self.eta_reg
        I = vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        inv_norm = 1.0 / (self._fft2K_norm2 + eta_reg)
        out = torch.zeros_like(I)
        for ch in range(self.channels):
            temp = torch.real(torch.fft.ifft2(torch.fft.fft2(I[:, ch:ch + 1, :, :]) * inv_norm))
            out[:, ch:ch + 1, :, :] = torch.real(torch.fft.ifft2(torch.fft.fft2(temp) * self._fft2K_conj))
        return out.reshape(vec.shape[0], -1)
