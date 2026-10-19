"""Solve linear inverse problems with FLINT (Stable Diffusion 3.5-Medium).

    python solve.py --config configs/ffhq/sr8.yaml --img_path samples/ffhq_example.png --workdir results/demo

Any entry of the config file can be overridden from the command line, e.g. `--stage2_steps 9`.
Results are written to <workdir>/{input,recon,label}/<image name>.png.
"""
import argparse
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from flint_sampler import FLINT
from functions.degradation import get_operator, measurement_image
from util import get_img_list, load_prompts, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='FLINT inverse problem solver')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--img_path', type=str, default=None, help='image file or folder')
    parser.add_argument('--prompt', type=str, default=None, help='text prompt used for every image')
    parser.add_argument('--prompt_file', type=str, default=None, help='one prompt per line, in image order')
    parser.add_argument('--workdir', type=str, default='results')
    parser.add_argument('--start_idx', type=int, default=0, help='first image index (for sharding)')
    parser.add_argument('--num_samples', type=int, default=-1, help='number of images (-1: all)')
    args, rest = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.img_path is not None:
        cfg['img_path'] = args.img_path
    if args.prompt is not None:
        cfg['prompt'], cfg['prompt_file'] = args.prompt, None
    elif args.prompt_file is not None:
        cfg['prompt'], cfg['prompt_file'] = None, args.prompt_file

    # remaining `--key value` pairs override config entries
    if len(rest) % 2:
        parser.error(f'unmatched arguments: {rest}')
    for key, val in zip(rest[::2], rest[1::2]):
        key = key.lstrip('-')
        if key not in cfg:
            parser.error(f'unknown option --{key}')
        cfg[key] = val if cfg[key] is None else type(cfg[key])(val)
    return args, cfg


def _tic():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    args, cfg = parse_args()
    workdir = Path(args.workdir)
    for sub in ('input', 'recon', 'label'):
        (workdir / sub).mkdir(parents=True, exist_ok=True)
    with open(workdir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    paths = get_img_list(Path(cfg['img_path']))
    end = len(paths) if args.num_samples < 0 else min(len(paths), args.start_idx + args.num_samples)
    indices = list(range(args.start_idx, end))
    prompts = load_prompts(cfg.get('prompt'), cfg.get('prompt_file'))
    prompt_of = lambda i: prompts[min(i, len(prompts) - 1)]

    flint = FLINT()
    embs, null = flint.encode_prompts([prompt_of(i) for i in indices])

    size = cfg['img_size']
    tf = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor()])
    if cfg['task'] != 'deblur_motion':
        A = get_operator(cfg['task'], cfg['deg_scale'], size, flint.device)

    for i in tqdm(indices, desc='FLINT'):
        path = paths[i]
        set_seed(cfg['seed'] + i)  # per-image seed: results do not depend on how images are sharded
        img = tf(Image.open(path).convert('RGB')).unsqueeze(0).to(flint.device) * 2 - 1
        if cfg['task'] == 'deblur_motion':
            A = get_operator(cfg['task'], cfg['deg_scale'], size, flint.device, index=i)

        y = A.A(img)
        y = y + cfg['sigma_y'] * torch.randn(y.shape, device=y.device,
                                             generator=torch.Generator(y.device).manual_seed(cfg['seed']))

        emb = embs[prompt_of(i)]
        t0 = _tic()
        x1 = flint.stage1(y, A, emb, null, cfg)              # proximal source initialization
        t1 = _tic()
        recon = flint.stage2(x1, y, A, emb, null, cfg)      # bounded-stochasticity posterior sampling
        t2 = _tic()
        tqdm.write(f'[{path.stem}] stage 1 {t1 - t0:.2f}s | stage 2 {t2 - t1:.2f}s | total {t2 - t0:.2f}s')

        save_image(measurement_image(cfg['task'], A, y, size), workdir / 'input' / f'{path.stem}.png')
        save_image(recon, workdir / 'recon' / f'{path.stem}.png')
        save_image((img / 2 + 0.5).clamp(0, 1), workdir / 'label' / f'{path.stem}.png')


if __name__ == '__main__':
    main()
