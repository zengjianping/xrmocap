# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim import build_optim_wrapper
from typing import Dict, List


def build_optimizer(model, cfg: Dict):
    optim_wrapper = build_optim_wrapper(model, cfg)
    return optim_wrapper.optimizer
