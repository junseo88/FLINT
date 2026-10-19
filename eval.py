"""Metrics used in the paper.

    python eval.py --path1 results/ffhq_sr8/label --path2 results/ffhq_sr8/recon --metric psnr ssim lpips fid

Images are paired by file name. `fid` is the patch-based FID: every 768x768 image is tiled into
nine non-overlapping 256x256 patches, and InceptionV3 pool features of the patches are compared.
"""
import argparse
from pathlib import Path
from typing import List

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from pytorch_msssim import ssim
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


def tag(name: str):
    def wrapper(func):
        func.tag = name
        return func
    return wrapper


class Factory(object):
    def __init__(self, name: List[str]):
        self.name = name
        methods = {func for func in dir(self) if callable(getattr(self, func)) and hasattr(getattr(self, func), 'tag')}
        self.tagged_method = {getattr(self, func).tag: getattr(self, func) for func in methods}
        self._call_func = self.get_method(name)

    def retrieve(self, input_dir, pred_dir):
        """Pairs of (input, pred) paths matched by file name."""
        exts = ('.png', '.jpg', '.jpeg')
        preds = {p.name: p for p in Path(pred_dir).iterdir() if p.suffix.lower() in exts}
        inputs = sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() in exts and p.name in preds)
        if not inputs:
            raise FileNotFoundError(f'no matching images in {input_dir} and {pred_dir}')
        return inputs, [preds[p.name] for p in inputs]

    def __call__(self, *args, **kwargs):
        return [_func(*args, **kwargs) for _func in self._call_func]

    def get_method(self, name: List[str]):
        methods = []
        for n in name:
            if n not in self.tagged_method:
                raise ValueError(f'Cannot find {self.__class__.__name__} ({n})')
            methods.append(self.tagged_method[n])
        return methods


class _Patches(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, x, y, ps = self.samples[idx]
        return self.transform(Image.open(path).convert('RGB').crop((x, y, x + ps, y + ps)))


class Metric(Factory):
    @tag('psnr')
    def _psnr(self, input_path, pred_path, data_range: int = 255, **kwargs):
        values = []
        for in_f, pred_f in zip(*self.retrieve(input_path, pred_path)):
            img1 = transforms.ToTensor()(Image.open(in_f).convert('RGB')).unsqueeze(0).cuda() * data_range
            img2 = transforms.ToTensor()(Image.open(pred_f).convert('RGB')).unsqueeze(0).cuda() * data_range
            mse = F.mse_loss(img1, img2)
            values.append(100.0 if mse == 0 else 10 * torch.log10(data_range ** 2 / mse).item())
        return np.mean(values)

    @tag('ssim')
    def _ssim(self, input_path, pred_path, data_range: int = 255, **kwargs):
        values = []
        for in_f, pred_f in zip(*self.retrieve(input_path, pred_path)):
            img1 = transforms.ToTensor()(Image.open(in_f).convert('RGB')).unsqueeze(0).cuda() * data_range
            img2 = transforms.ToTensor()(Image.open(pred_f).convert('RGB')).unsqueeze(0).cuda() * data_range
            values.append(ssim(img1, img2, data_range=data_range, size_average=True).item())
        return np.mean(values)

    @tag('lpips')
    def _lpips(self, input_path, pred_path, **kwargs):
        lpips_fn = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
        values = []
        with torch.no_grad():
            for in_f, pred_f in zip(*self.retrieve(input_path, pred_path)):
                img1 = transforms.ToTensor()(Image.open(in_f).convert('RGB')).cuda()
                img2 = transforms.ToTensor()(Image.open(pred_f).convert('RGB')).cuda()
                values.append(lpips_fn(img1, img2).item())
        return np.mean(values)

    @tag('fid')
    def _fid(self, input_path, pred_path, patch_size: int = 256, batch_size: int = 32, **kwargs):
        in_fs, pred_fs = self.retrieve(input_path, pred_path)
        weights = torchvision.models.Inception_V3_Weights.DEFAULT
        model = torchvision.models.inception_v3(weights=weights).eval().cuda()
        extractor = create_feature_extractor(model, return_nodes={'avgpool': 'feat'})

        def tiles(path):
            with Image.open(path) as im:
                W, H = im.size
            xs = list(range(0, W - patch_size + 1, patch_size))
            ys = list(range(0, H - patch_size + 1, patch_size))
            if xs[-1] != W - patch_size:
                xs.append(W - patch_size)
            if ys[-1] != H - patch_size:
                ys.append(H - patch_size)
            return [(x, y) for y in ys for x in xs]

        coords = [tiles(p) for p in in_fs]  # patch grid taken from the reference images

        @torch.no_grad()
        def features(paths):
            samples = [(p, x, y, patch_size) for p, c in zip(paths, coords) for (x, y) in c]
            loader = DataLoader(_Patches(samples, weights.transforms()), batch_size=batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
            feats = [extractor(x.cuda(non_blocking=True))['feat'].squeeze(-1).squeeze(-1).cpu() for x in loader]
            return torch.cat(feats).numpy().astype(np.float64)

        f1, f2 = features(in_fs), features(pred_fs)
        mu1, mu2 = f1.mean(0), f2.mean(0)
        s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
        covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(s1.shape[0]) * 1e-6
            covmean, _ = linalg.sqrtm((s1 + offset).dot(s2 + offset), disp=False)
        covmean = covmean.real
        diff = mu1 - mu2
        return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=Path, help='ground-truth images (e.g. <workdir>/label)')
    parser.add_argument('--path2', type=Path, help='reconstructions (e.g. <workdir>/recon)')
    parser.add_argument('--metric', type=str, nargs='+', default=['psnr', 'ssim', 'lpips', 'fid'])
    args = parser.parse_args()

    metric = Metric(args.metric)
    output = metric(args.path1, args.path2)
    for m, o in zip(args.metric, output):
        print(f'{m}: {o:.4f}')
