## from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
# from torchvision.models import ResNet
# from .resnet_imagenet import ResNet
import pudb


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


## Modified from https://github.com/asdf2kr/BAM-CBAM-pytorch/
def conv7x7(in_channels, out_channels, stride=1, padding=3, dilation=1):
    ''' 7x7 convolution '''
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=padding, dilation=dilation, bias=False)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, num_classes=10, dilation=1):
        super(CBAMLayer, self).__init__()
        self.hid_channel = channel // reduction
        self.dilation = dilation

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.globalMaxPool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP.
        self.mlp = nn.Sequential(
            nn.Linear(in_features=channel, out_features=self.hid_channel),
            nn.ReLU(),
            nn.Linear(in_features=self.hid_channel, out_features=channel)
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = conv7x7(2, 1, stride=1, dilation=self.dilation)
        self.classifier_route = nn.Linear(channel, num_classes)

    def forward(self, x):
        ''' Channel attention '''
        avgOut = self.globalAvgPool(x)
        avgOut = avgOut.view(avgOut.size(0), -1)
        avgOut = self.mlp(avgOut)

        maxOut = self.globalMaxPool(x)
        maxOut = maxOut.view(maxOut.size(0), -1)
        maxOut = self.mlp(maxOut)
        # sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))
        Mc = self.sigmoid(avgOut + maxOut)

        Mc1 = Mc.view(Mc.size(0), Mc.size(1), 1, 1)
        Mf1 = Mc1 * x

        # print(Mc.shape)
        # print(self.classifier_route)

        route_classifier_out = self.classifier_route(Mc)

        ''' Spatial attention. '''
        # sigmoid(conv7x7( [AvgPool(F); MaxPool(F)]))
        maxOut = torch.max(Mf1, 1)[0].unsqueeze(1)
        avgOut = torch.mean(Mf1, 1).unsqueeze(1)
        Ms = torch.cat((maxOut, avgOut), dim=1)

        Ms = self.conv1(Ms)
        Ms = self.sigmoid(Ms)
        Ms = Ms.view(Ms.size(0), 1, Ms.size(2), Ms.size(3))
        Mf2 = Ms * Mf1

        # return Mf2, torch.Tensor(0).to(x.device), torch.Tensor(0).to(x.device)
        return Mf2, Mc, route_classifier_out



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, num_classes=10):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.classifier_route = nn.Linear(channel, num_classes)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        y = self.fc(y).view(b, c)

        route_classifier_out = self.classifier_route(y)

        return x * y.view(b, c, 1, 1).expand_as(x), y, route_classifier_out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, num_classes=1000, num_classes_route=1000):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction, num_classes=num_classes_route)
        self.downsample = downsample
        self.stride = stride

    def forward(self, xr):
        x = xr[0]
        sc = xr[1]
        route_classifier_outs = xr[2]

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out, sc_curr, route_classifier_outs_curr = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        sc = torch.cat((sc, sc_curr), 1)
        route_classifier_outs.append(route_classifier_outs_curr)

        return out, sc, route_classifier_outs


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16, num_classes=1000, num_classes_route=1000):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction, num_classes=num_classes_route)
        self.downsample = downsample
        self.stride = stride

    def forward(self, xr):
        x = xr[0]
        sc = xr[1]
        route_classifier_outs = xr[2]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out, sc_curr, route_classifier_outs_curr = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        sc = torch.cat((sc, sc_curr), 1)
        route_classifier_outs.append(route_classifier_outs_curr)

        return out, sc, route_classifier_outs



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=20, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, num_classes_route=20):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_classes=num_classes, num_classes_route=num_classes_route)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], num_classes=num_classes, num_classes_route=num_classes_route)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], num_classes=num_classes, num_classes_route=num_classes_route)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], num_classes=num_classes, num_classes_route=num_classes_route)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SEBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, SEBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, num_classes=1000, num_classes_route=1000):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, num_classes=num_classes, num_classes_route=num_classes_route))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, num_classes=num_classes, num_classes_route=num_classes_route))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        route_outs = torch.Tensor(0).to(x.device)
        route_classifier_outs = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        xr = [x, route_outs, route_classifier_outs]

        xr = self.layer1(xr)
        xr = self.layer2(xr)
        xr = self.layer3(xr)
        xr = self.layer4(xr)

        x, route_outs, route_classifier_outs = xr[0], xr[1], xr[2]

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, route_outs, route_classifier_outs

    def forward(self, x):
        return self._forward_impl(x)


def se_resnet18(num_classes=1000, num_classes_route=1000):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_classes_route=num_classes_route)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1000, num_classes_route=1000):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, num_classes_route=num_classes_route)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1000, num_classes_route=1000, pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, num_classes_route=num_classes_route)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


def se_resnet101(num_classes=1000, num_classes_route=1000):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, num_classes_route=num_classes_route)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes=1000, num_classes_route=1000):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes, num_classes_route=num_classes_route)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class CifarSE_PRE_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSE_PRE_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(inplanes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        # print(x[0].shape)
        # print(x[1].shape)
        residual = self.downsample(x)
        out = self.se(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class CifarSE_POST_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSE_POST_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        # print(x[0].shape)
        # print(x[1].shape)
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        out = self.se(out)

        return out



class CifarSE_POST_Reversed_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSE_POST_Reversed_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(inplanes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        # print(x[0].shape)
        # print(x[1].shape)
        out = self.se(x)

        residual = self.downsample(out)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16, num_classes_route=10, attention_type="SE"):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction, num_classes=num_classes_route)
        self.cbam = CBAMLayer(planes, reduction, num_classes=num_classes_route)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride
        self.attention_type = attention_type

    def forward(self, xr):
        # print(x_in[0].shape)
        # print(x_in[1].shape)
        x = xr[0]
        sc = xr[1]
        route_classifier_outs = xr[2]

        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.attention_type=="SE":
            x, sc_curr, route_classifier_outs_curr = self.se(x)

            sc = torch.cat((sc, sc_curr), 1)
            route_classifier_outs.append(route_classifier_outs_curr)
        elif self.attention_type=="CBAM":
            x, sc_curr, route_classifier_outs_curr = self.cbam(x)

            sc = torch.cat((sc, sc_curr), 1)
            route_classifier_outs.append(route_classifier_outs_curr)

        # sc = torch.cat((sc, sc_curr), 1)
        # route_classifier_outs.append(route_classifier_outs_curr)

        x += residual
        x = self.relu(x)

        return x, sc, route_classifier_outs


class CifarSEResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, width_mult=1., reduction=16, num_classes_route=10, attention_type="SE"):
        super(CifarSEResNet, self).__init__()
        print('width_mult: ', width_mult)

        # self.inplane = 16
        round_nearest = 8
        self.inplane = _make_divisible(16 * width_mult, round_nearest)
        print('self.inplane: ', self.inplane)

        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, _make_divisible(16 * width_mult, round_nearest), blocks=n_size, stride=1, reduction=reduction
            , num_classes_route=num_classes_route, attention_type=attention_type)
        self.layer2 = self._make_layer(
            block, _make_divisible(32 * width_mult, round_nearest), blocks=n_size, stride=2, reduction=reduction
            , num_classes_route=num_classes_route, attention_type=attention_type)
        self.layer3 = self._make_layer(
            block, _make_divisible(64 * width_mult, round_nearest), blocks=n_size, stride=2, reduction=reduction
            , num_classes_route=num_classes_route, attention_type=attention_type)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(_make_divisible(64 * width_mult, round_nearest), num_classes)
        self.initialize()

        # self.pruning = pruning
        # self.pruning_ratio = pruning_ratio

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction, num_classes_route, attention_type):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction, num_classes_route=num_classes_route, attention_type=attention_type))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # pu.db
        route_outs = torch.Tensor(0).to(x.device)
        route_classifier_outs = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        xr = [x, route_outs, route_classifier_outs]

        xr = self.layer1(xr)
        xr = self.layer2(xr)
        xr = self.layer3(xr)

        x, route_outs, route_classifier_outs = xr[0], xr[1], xr[2]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, route_outs, route_classifier_outs


class CifarSEPreActResNet(CifarSEResNet):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEPreActResNet, self).__init__(
            block, n_size, num_classes, reduction)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


def se_pre_resnet20(**kwargs):
    """Constructs a ResNet-18 model. with SE-PRE

    """
    model = CifarSEResNet(CifarSE_PRE_BasicBlock, 3, **kwargs)
    return model


def se_post_resnet20(**kwargs):
    """Constructs a ResNet-18 model. with SE-PRE

    """
    model = CifarSEResNet(CifarSE_POST_BasicBlock, 3, **kwargs)
    return model


def se_post_reversed_resnet20(**kwargs):
    """Constructs a ResNet-18 model. with SE-PRE

    """
    model = CifarSEResNet(CifarSE_POST_Reversed_BasicBlock, 3, **kwargs)
    return model


def se_resnet20(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_resnet32(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_resnet56(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


def se_resnet44(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 7, **kwargs)
    return model


def se_resnet110(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 18, **kwargs)
    return model


def se_preactresnet20(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_preactresnet32(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_preactresnet56(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 9, **kwargs)
    return model
