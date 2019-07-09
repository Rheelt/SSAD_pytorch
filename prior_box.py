import torch
import torch.nn as nn
import numpy as np
from config import Config


class PriorBox(nn.Module):
    def __init__(self, config):
        super(PriorBox, self).__init__()
        self.layer_names = config.layer_names
        self.num_cells = config.num_cells
        self.base_scale = config.base_scale
        self.aspect_ratios = config.aspect_ratios
        self.priors_center = {}
        self.priors_width = {}
        self._generating_box()

    def _generating_box(self):
        """Generate SSAD Prior Boxes.
                """
        for layer_name, layer_step, scale, ratios in zip(self.layer_names, self.num_cells, self.base_scale,
                                                         self.aspect_ratios):
            width_set = [scale * ratio for ratio in ratios]
            center_set = [1. / layer_step * i + 0.5 / layer_step for i in range(layer_step)]
            width_default = []
            center_default = []
            for i in range(layer_step):
                for j in range(len(ratios)):
                    width_default.append(width_set[j])
                    center_default.append(center_set[i])
            width_default = np.array(width_default).reshape(1, -1)
            center_default = np.array(center_default).reshape(1, -1)
            width_default = torch.Tensor(width_default)
            center_default = torch.Tensor(center_default)
            self.priors_center.setdefault(layer_name, center_default)
            self.priors_width.setdefault(layer_name, width_default)

    def forward(self, output_name):
        return self.priors_center[output_name], self.priors_width[output_name]


if __name__ == '__main__':
    config = Config()
    priorBox = PriorBox(config)
    priorBox('AL1')
