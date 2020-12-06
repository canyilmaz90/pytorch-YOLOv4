from ..utils import *
from tool.torch_utils import *
from tool.yolo_layer import YoloLayer


class PartialBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(in_channels, in_channels, 3, 1, 'leaky')
        # Here [route] with groups applies
        self.conv2 = Conv_Bn_Activation(in_channels//2, in_channels//2, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(in_channels//2, in_channels//2, 3, 1, 'leaky')
        # Here goes concatenation
        self.conv4 = Conv_Bn_Activation(in_channels, in_channels, 1, 1, 'leaky')
        # It's concatenation again
        self.max1 = nn.MaxPool2d(2, 2)

    def forward(self, inp, feat_out=False):
        x1 = self.conv1(inp)
        g0, g1 = torch.split(x1, [x1.shape[1]//2, x1.shape[1]//2], dim=1)
        x2 = self.conv2(g1)
        x3 = self.conv3(x2)
        c1 = torch.cat([x3, x2], dim=1)
        x4 = self.conv4(c1)
        c2 = torch.cat([x1, x4], dim=1)
        out = self.max1(c2)
        if feat_out:
            return out, x4
        return out


class Yolov4TinyHead(nn.Module):
    def __init__(self, n_classes, inference):
        super().__init__()
        self.inference = inference
        self.conv1 = Conv_Bn_Activation(512, 512, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv4 = Conv_Bn_Activation(512, (4 + 1 + n_classes) * 3, 1, 1, 'linear', bn=False, bias=True)
        self.yolo1 = YoloLayer(
                                anchor_mask=[3, 4, 5], num_classes=n_classes,
                                anchors=[10, 14,  23, 27,  37, 58,  81, 82,  135, 169,  344, 319],
                                num_anchors=6, stride=32)
        self.conv5 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.upsample1 = Upsample()
        self.conv6 = Conv_Bn_Activation(384, 256, 3, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, (4 + 1 + n_classes) * 3, 1, 1, 'linear', bn=False, bias=True)
        self.yolo2 = YoloLayer(
                                anchor_mask=[1, 2, 3], num_classes=n_classes,
                                anchors=[10, 14,  23, 27,  37, 58,  81, 82,  135, 169,  344, 319],
                                num_anchors=6, stride=16)

    def forward(self, inp, feat_in):
        x1 = self.conv1(inp)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # route to 27 (x2)
        x5 = self.conv5(x2)
        shape = [int(x5.shape[2] * 2), int(x5.shape[3] * 2)]
        up = self.upsample1(x5, shape)
        c1 = torch.cat([up, feat_in], dim=1)
        x6 = self.conv6(c1)
        x7 = self.conv7(x6)

        if self.inference:
            y1 = self.yolo1(x4)
            y2 = self.yolo2(x7)
            return get_region_boxes([y1, y2])
        else:
            return [x4, x7]


class Yolov4Tiny(nn.Module):
    def __init__(self, n_classes=80, inference=False):
        super().__init__()

        self.conv1 = Conv_Bn_Activation(3, 32, 3, 2, 'leaky')
        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'leaky')
        self.partial1 = PartialBlock(64)
        self.partial2 = PartialBlock(128)
        self.partial3 = PartialBlock(256)
        self.head = Yolov4TinyHead(n_classes, inference)

    def forward(self, inp):
        d1 = self.conv1(inp)
        d2 = self.conv2(d1)
        d3 = self.partial1(d2)
        d4 = self.partial2(d3)
        d5, c23 = self.partial3(d4, feat_out=True)
        out = self.head(d5, c23)
        return out
