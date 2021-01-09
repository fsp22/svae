from typing import Any

import torch


class Loss(torch.nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, anneal, *model_output):
        raise NotImplementedError('Loss class should be overwritten!')
