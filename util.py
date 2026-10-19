import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_img_list(root: Path):
    if root.is_dir():
        return (sorted(root.glob('*.png')) + sorted(root.glob('*.jpg')) + sorted(root.glob('*.jpeg')))
    return [root]


def load_prompts(prompt=None, prompt_file=None):
    """A single prompt for every image, or one prompt per line of `prompt_file` (in image order)."""
    if prompt is not None:
        return [prompt]
    if prompt_file is not None:
        with open(prompt_file) as f:
            return [line.rstrip('\n') for line in f]
    raise ValueError('either `prompt` or `prompt_file` must be given')
