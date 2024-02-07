import torch
import torch.nn as nn



class PointCloudTaskModel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(PointCloudTaskModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out

    def set_mode(self, mode="fine_tuning"):
        """
        Set the mode of the model: 'fine_tuning' or 'linear_probing'.
        """
        if mode == "fine_tuning":
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
        elif mode == "linear_probing":
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            raise ValueError("Invalid mode selected. Choose 'fine_tuning' or 'linear_probing'.")
