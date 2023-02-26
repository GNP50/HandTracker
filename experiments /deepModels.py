import torch
import torch.nn as nn


class DeepFeedForward(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.deep_layers = []
        self.input_shape = kwargs["input_shape"]
        self.deep = kwargs["deep"]
        self.output_shapes = kwargs["output_shapes"]
        self.task = kwargs["task"]
        self.classes = kwargs["classes"]

        in_ = kwargs["input_shape"]

        for i in range(0, kwargs["deep"]):
            self.deep_layers.append(nn.Linear(
                in_features=in_, out_features=self.output_shapes[i]
            ))
            in_ = self.output_shapes[i]

        self.last = nn.Linear(
            in_features=in_, out_features=self.classes
        )

        self.deep_layers = nn.ModuleList(self.deep_layers)

    def forward(self, features):
        for layer in self.deep_layers:
            features = layer(features)
            features = nn.Tanh()(features)
        features = self.last(features)
        features = nn.Softmax(dim=1)(features)

        return features
