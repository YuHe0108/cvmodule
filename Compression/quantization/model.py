from torch import nn


class VGG_prunable(nn.Module):
    def __init__(self, cfg):
        super(VGG_prunable, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(cfg[-2], 10)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def VGG_11_prune(cfg=None):
    if cfg is None:
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG_prunable(cfg)


if __name__ == '__main__':
    print(VGG_11_prune())
