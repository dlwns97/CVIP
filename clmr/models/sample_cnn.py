import torch
import torch.nn as nn
from .model import Model


class SampleCNN(Model):
    def __init__(self, strides, supervised, out_dim):
        super(SampleCNN, self).__init__()

        self.strides = strides
        self.supervised = supervised
        self.sequential = [
            nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        ]

        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 512],
        ]

        assert len(self.hidden) == len(
            self.strides
        ), "Number of hidden layers and strides are not equal"
        for stride, (h_in, h_out) in zip(self.strides, self.hidden):
            self.sequential.append(
                nn.Sequential(
                    nn.Conv1d(h_in, h_out, kernel_size=stride, stride=1, padding=1),
                    nn.BatchNorm1d(h_out),
                    nn.ReLU(),
                    nn.MaxPool1d(stride, stride=stride),
                )
            )

        # 1 x 512
        self.sequential.append(
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
        )

        self.sequential = nn.Sequential(*self.sequential)

        if self.supervised:
            self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        out = self.sequential(x)
        if self.supervised:
            out = self.dropout(out)

        out = out.reshape(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)
        return logit

class SampleCNN_1550(nn.Module):
    def __init__(self, strides, supervised, out_dim):
        super(SampleCNN_1550, self).__init__()

        self.supervised = supervised

        # 첫 번째 계층
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # 다음 계층
        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 512],
        ]

        assert len(self.hidden) == len(strides), "Number of hidden layers and strides are not equal"

        for i, (stride, (h_in, h_out)) in enumerate(zip(strides, self.hidden)):
            setattr(self, f'conv{i+2}', nn.Sequential(
                nn.Conv1d(h_in, h_out, kernel_size=stride, stride=1, padding=1),
                nn.BatchNorm1d(h_out),
                nn.ReLU(),
                nn.MaxPool1d(stride, stride=stride),
            ))

        # 마지막 Conv1d 계층
        self.conv11 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        if self.supervised:
            self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(2, 12):  # conv2부터 conv11까지
            x = getattr(self, f'conv{i}')(x)

        if self.supervised:
            x = self.dropout(x)

        x = x.reshape(x.shape[0], -1)
        logit = self.fc(x)
        return logit
