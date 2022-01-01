import torch
import torch.nn as nn


class ModelMonitor(nn.Module):
    def __init__(self, config=None) -> None:
        super().__init__()
        self.config = {}
        self.config["input_size"] = 51200
        self.config["in_channels"] = 1
        self.config["out_channels"] = 64
        self.config["kernel_size"] = 64
        self.config["stride"] = 64
        self.config["padding"] = 0
        self.config["dropout_p"] = 0.3

        self.config = self._update_config(
            default_config=self.config, user_config=config)

        input_size = self.config["input_size"]
        in_channels = self.config["in_channels"]
        out_channels = self.config["out_channels"]
        kernel_size = self.config["kernel_size"]
        stride = self.config["stride"]
        padding = self.config["padding"]
        dropout_p = self.config["dropout_p"]

        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        shpae_out_conv = self._cal_dim(
            input_size=input_size, module_list=[self.conv1])

        self.fc1 = nn.Linear(shpae_out_conv[1]*shpae_out_conv[2], 64)
        self.fc2 = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    @ staticmethod
    def _cal_dim(input_size, module_list):
        x = torch.randn((2, 1, input_size))
        for m in module_list:
            x = m(x)
        return x.shape

    @ staticmethod
    def _update_config(default_config, user_config):
        if user_config is None:
            return default_config
        for key in user_config.keys():
            if key in default_config.keys():
                default_config[key] = user_config[key]
        return default_config


if __name__ == "__main__":
    config = {}
    config["out_channels"] = 32
    model = ModelMonitor(config=config)

    batch_size = 2
    input_channels = 1
    input_size = 51200

    x = torch.randn((batch_size, input_channels, input_size))
    out = model(x)
    print(out.shape)
