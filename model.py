import torch
from torch.nn import init
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from config import Config
from prior_box import PriorBox


class SSAD(nn.Module):
    def __init__(self, config):
        super(SSAD, self).__init__()
        self.num_classes = config.num_classes
        self.num_anchors = config.num_anchors
        self.input_feature_dim = config.feature_dim
        self.prediction_output = self.num_anchors * (self.num_classes + 3)
        self.best_loss = 10000000
        self.prior_box = PriorBox(config)
        # Base Layers
        self.base_layers = nn.Sequential(OrderedDict([
            ('conv1d_1',
             nn.Conv1d(in_channels=self.input_feature_dim, out_channels=512, kernel_size=9, stride=1, padding=4)),
            ('relu_1', nn.ReLU()),
            ('maxpooling1d_1', nn.MaxPool1d(kernel_size=4, stride=2, padding=1)),
            ('conv1d_2', nn.Conv1d(in_channels=512, out_channels=512, kernel_size=9, stride=1, padding=4)),
            ('relu_2', nn.ReLU()),
            ('maxpooling1d_2', nn.MaxPool1d(kernel_size=4, stride=2, padding=1))
        ]))

        # Anchor Layers
        self.anchor_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.anchor_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
        self.anchor_layer3 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU())

        # Prediction Layers
        self.prediction_layer1 = nn.Conv1d(in_channels=1024, out_channels=self.prediction_output, kernel_size=3,
                                           stride=1, padding=1)
        self.prediction_layer2 = nn.Conv1d(in_channels=1024, out_channels=self.prediction_output, kernel_size=3,
                                           stride=1, padding=1)
        self.prediction_layer3 = nn.Conv1d(in_channels=1024, out_channels=self.prediction_output, kernel_size=3,
                                           stride=1, padding=1)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv1d):
            init.xavier_uniform_(m.weight)
            # init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, input, device):
        """
        Forward pass logic
        :return: Model output
        """
        base_feature = self.base_layers(input)

        anchor1 = self.anchor_layer1(base_feature)
        anchor2 = self.anchor_layer2(anchor1)
        anchor3 = self.anchor_layer3(anchor2)

        prediction1 = self.prediction_layer1(anchor1)
        prediction2 = self.prediction_layer1(anchor2)
        prediction3 = self.prediction_layer1(anchor3)

        batch_size = prediction1.shape[0]

        prediction1 = prediction1.view(batch_size, -1, prediction1.shape[-1] * self.num_anchors)
        prediction2 = prediction2.view(batch_size, -1, prediction2.shape[-1] * self.num_anchors)
        prediction3 = prediction3.view(batch_size, -1, prediction3.shape[-1] * self.num_anchors)

        prediction1_x = prediction1[:, -2, :]
        prediction1_w = prediction1[:, -1, :]
        prediction1_x = prediction1_x * self.prior_box('AL1')[1].to(device) * 0.1 + self.prior_box('AL1')[0].to(device)
        prediction1_w = torch.exp(0.1 * prediction1_w) * self.prior_box('AL1')[1].to(device)
        prediction1_score = prediction1[:, -3, :]
        prediction1_score = torch.sigmoid(prediction1_score)
        prediction1_label = prediction1[:, :self.num_classes, :]

        prediction2_x = prediction2[:, -2, :]
        prediction2_w = prediction2[:, -1, :]
        prediction2_x = prediction2_x * self.prior_box('AL2')[1].to(device) * 0.1 + self.prior_box('AL2')[0].to(device)
        prediction2_w = torch.exp(0.1 * prediction2_w) * self.prior_box('AL2')[1].to(device)
        prediction2_score = prediction2[:, -3, :]
        prediction2_score = torch.sigmoid(prediction2_score)
        prediction2_label = prediction2[:, :self.num_classes, :]

        prediction3_x = prediction3[:, -2, :]
        prediction3_w = prediction3[:, -1, :]
        prediction3_x = prediction3_x * self.prior_box('AL3')[1].to(device) * 0.1 + self.prior_box('AL3')[0].to(device)
        prediction3_w = torch.exp(0.1 * prediction3_w) * self.prior_box('AL3')[1].to(device)
        prediction3_score = prediction3[:, -3, :]
        prediction3_score = torch.sigmoid(prediction3_score)
        prediction3_label = prediction3[:, :self.num_classes, :]

        all_prediction_x = torch.cat((prediction1_x, prediction2_x, prediction3_x), dim=-1)
        all_prediction_w = torch.cat((prediction1_w, prediction2_w, prediction3_w), dim=-1)
        all_prediction_score = torch.cat((prediction1_score, prediction2_score, prediction3_score), dim=-1)
        all_prediction_label = torch.cat((prediction1_label, prediction2_label, prediction3_label), dim=-1)

        return all_prediction_x, all_prediction_w, all_prediction_score, all_prediction_label


if __name__ == '__main__':
    config = Config()
    model = SSAD(config)
    input = torch.Tensor(np.zeros(shape=(4, 3072, 128)))
    model(input)
