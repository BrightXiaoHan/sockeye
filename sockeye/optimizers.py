# Copyright 2017--2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

import torch

from . import config
from . import constants as C

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig(config.Config):
    # Optimizer
    name: str

    # Adam default values
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.

    # SGD default value
    momentum: float = 0.

    # Applied outside of optimizer
    gradient_clipping_type: str = C.GRADIENT_CLIPPING_TYPE_NONE
    gradient_clipping_threshold: Optional[float] = None
    update_interval: int = 1


def get_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """
    Create an optimizer for a Sockeye model using the specified config settings.

    :param model: Sockeye model.
    :param config: Optimizer config.

    :return: Optimizer.
    """
    if config.name == C.OPTIMIZER_ADAM:
        return torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps,
                                weight_decay=config.weight_decay)
    elif config.name == C.OPTIMIZER_SGD:
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                               weight_decay=config.weight_decay)
    raise ValueError(f'Unknown optimizer: {config.name}')
