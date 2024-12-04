import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torch.nn.parameter import Parameter
import math


model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        
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
        
        ##
        self.conv0 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        ##
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(512 * block.expansion, num_classes)


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
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
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
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        device = x.device
        x = self.conv0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_out(x)

        return x


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[BasicBlock],
    layers: List[int],
    pretrained: bool,
    pretrained_path: str,
    **kwargs: Any
) -> ResNet:
    
    model = ResNet(block, layers, **kwargs)
    model_dict = model.state_dict()
    if pretrained:
        state_dict = torch.load(pretrained_path)
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=True)
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model


def resnet34(pretrained: bool = False, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, **kwargs)


class InvResX1D(nn.Module):
    """
    Inverted Residual Block 1D - ConvNeXt style. Applies LayerNorm on the channel dimension (suited for temporal data
    where D is identified as time)
    """

    def __init__(self, indim, outdim, kernel, stride=1, expansion_fact=4, groups=1):
        super(InvResX1D, self).__init__()
        self.depth_wise = nn.Conv1d(indim, indim, kernel, stride=stride, padding=int(0.5 * (kernel - 1)),
                                    groups=indim)
        if groups > 1:
            self.norm = nn.GroupNorm(groups, indim)
        else:
            self.norm = nn.LayerNorm(indim)
        self.pt_wise_in = nn.Conv1d(indim, expansion_fact * indim, 1, groups=groups)
        self.act = nn.GELU()
        self.pt_wise_out = nn.Conv1d(expansion_fact * indim, outdim, 1, groups=groups)
        
        self.downsample = None
        if (stride != 1) or (indim != outdim):
            self.downsample = nn.Conv1d(indim, outdim, 1, stride, groups=groups)
            if groups > 1:
                self.out_norm = nn.GroupNorm(groups, outdim)
            else:
                self.out_norm = nn.LayerNorm(outdim)
        self.groups = groups

    # def norm_out(self, x, norm_list):
    #     in_dim = x.shape[1]
    #     splits = x.transpose(1, 2).split(int(in_dim / self.groups), dim=-1)
    #     return torch.cat([norm_list[i](split) for (i, split) in enumerate(splits)], dim=-1).transpose(1, 2)
        
    def forward(self, x):
        # Expected shape: B x C x D, resp batch size, channels, dimension
        identity = x
        out = self.depth_wise(x)
        if self.groups > 1:
            out = self.norm(out)
        else:
            out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.pt_wise_in(out)
        out = self.act(out)
        out = self.pt_wise_out(out)
        
        if self.downsample is not None:
            if self.groups > 1:
                identity = self.out_norm(self.downsample(x))
            else:
                identity = self.out_norm(self.downsample(x).transpose(1, 2)).transpose(1, 2)
            
        out += identity
        out = self.act(out)
            
        return out
    
    
class InvResX2D(nn.Module):
    """
    Inverted Residual Block 2D - ConvNeXt style
    """

    def __init__(self, indim, outdim, kernel, stride=1, expansion_fact=4):
        super(InvResX2D, self).__init__()
        if type(kernel) == tuple:
            padding = (int(0.5 * (kernel[0] - 1)), int(0.5 * (kernel[1] - 1)))
        else:
            padding = int(0.5 * (kernel - 1))
        self.depth_wise = nn.Conv2d(indim, indim, kernel, stride=stride, padding=padding,
                                    groups=indim)
        self.norm = nn.BatchNorm2d(indim)
        self.pt_wise_in = nn.Conv2d(indim, expansion_fact * indim, 1)
        self.act = nn.GELU()
        self.pt_wise_out = nn.Conv2d(expansion_fact * indim, outdim, 1)
                          
        self.downsample = None
        if (stride != 1) or (indim != outdim):
            self.downsample = nn.Sequential(
                nn.Conv2d(indim, outdim, 1, stride),
                nn.BatchNorm2d(outdim)
            )
                          
        
    def forward(self, x):
        # Expected shape: B x C x H x W
        identity = x
        out = self.depth_wise(x)
        out = self.norm(out)
        out = self.pt_wise_in(out)
        out = self.act(out)
        out = self.pt_wise_out(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.act(out)
            
        return out


class InvResX3D(nn.Module):
    """
    Inverted Residual Block 3D - ConvNeXt style
    """

    def __init__(self, indim, outdim, kernel, stride=1, expansion_fact=4):
        super(InvResX3D, self).__init__()
        if type(kernel) == tuple:
            padding = (int(0.5 * (kernel[0] - 1)), int(0.5 * (kernel[1] - 1)))
        else:
            padding = int(0.5 * (kernel - 1))
        self.depth_wise = nn.Conv3d(indim, indim, kernel, stride=stride, padding=padding,
                                    groups=indim)
        self.norm = nn.BatchNorm3d(indim)
        self.pt_wise_in = nn.Conv3d(indim, expansion_fact * indim, 1)
        self.act = nn.GELU()
        self.pt_wise_out = nn.Conv3d(expansion_fact * indim, outdim, 1)
                          
        self.downsample = None
        if (stride != 1) or (indim != outdim):
            self.downsample = nn.Sequential(
                nn.Conv3d(indim, outdim, 1, stride),
                nn.BatchNorm3d(outdim)
            )
                          
        
    def forward(self, x):
        # Expected shape: B x C x H x W
        identity = x
        out = self.depth_wise(x)
        out = self.norm(out)
        out = self.pt_wise_in(out)
        out = self.act(out)
        out = self.pt_wise_out(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.act(out)
            
        return out


class ConvNeXt(nn.Module):
    
    def __init__(self, inplanes, block, layers, out_dim, sa=False, avgpool=1):
        super(ConvNeXt, self).__init__()
        
        if block.__name__ == 'InvResX1D':
            dim = 1
            self.kernel = 3
            self.avgpool = nn.AdaptiveAvgPool1d(avgpool)
        elif block.__name__ == 'InvResX2D':
            dim = 2
            # self.kernel = (7, 3)
            # self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
            self.kernel = 3
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d(avgpool)
        elif block.__name__ == 'InvResX3D':
            dim = 3
            self.kernel = 3
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AdaptiveAvgPool3d(avgpool)
        
        self.inplanes = max(inplanes, 32)
        self.conv1 = block(inplanes, self.inplanes, self.kernel)
        if sa:
            self.sa = SelfAttention(self.inplanes, self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        
        self.fc_out = nn.Linear(512 * (avgpool ** dim), out_dim)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=2):
        layers = []
        planes = max(planes, self.inplanes) if planes != 512 else planes
        layers.append(block(self.inplanes, planes, self.kernel, stride))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, self.kernel))
        self.inplanes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)

        if hasattr(self, 'sa'):
            x = self.sa(x)
        if hasattr(self, 'maxpool'):
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_out(x)
        
        return x


class DeepMLP(nn.Module):
    def __init__(self, indim, outdim, layers=[1, 1, 1, 1], expansion_fact=1, groups=1):
        super(DeepMLP, self).__init__()
        
        dims = [(max(int(indim / (2 ** i)), outdim), max(int(indim / (2 ** (i + 1))), outdim)) 
            if i > 0 else (indim, max(int(indim / (2 ** (i + 1))), outdim)) for i in range(len(layers))]
        self.blocks = []
        for nblocks, (i, j) in zip(layers, dims):
            layer = [InvResX1D(i, i, kernel=1, expansion_fact=expansion_fact, groups=groups) for _ in range(nblocks - 1)]
            layer += [InvResX1D(i, j, kernel=1, expansion_fact=expansion_fact, groups=groups)]
            self.blocks += layer
        self.blocks = nn.Sequential(*self.blocks)
        if groups > 1:
            self.fc_out = nn.Conv1d(max(int(indim / (2 ** len(layers))), outdim), outdim, 1, groups=groups)
        else:
            self.fc_out = nn.Linear(max(int(indim / (2 ** len(layers))), outdim), outdim)
        self.groups = groups
        
    def forward(self, x):
        if self.groups > 1:
            return self.fc_out(self.blocks(x[..., None])).squeeze(-1)
        else:
            return self.fc_out(self.blocks(x[..., None]).squeeze(-1))


class FeaturePyramidNetwork(nn.Module):
    
    def __init__(self, feature_dim, interm_dim, n_layers=4, bottom_up_only=False):
        super(FeaturePyramidNetwork, self).__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv0 = nn.Sequential(*[
            InvResX1D(26, 64, 3),
            InvResX1D(64, 128, 3, 2),
            InvResX1D(128, 256, 3),
            InvResX1D(256, interm_dim, 3, 2)
        ])
        if bottom_up_only:
            self.pyramid_layers = nn.ModuleList([
                InvResX1D(interm_dim, interm_dim, 7, 2) for _ in range(n_layers - 1)
            ])
        else:
            self.pyramid_layers = nn.ModuleList([
                nn.Sequential(*[
                    InvResX1D(interm_dim, interm_dim, 3),
                    InvResX1D(interm_dim, interm_dim, 3, 2) # à remplacer év par un seul layer avec k=7
                ]) for _ in range(n_layers - 1)
            ])
            self.p_convolutions = nn.ModuleList([
                nn.Conv1d(in_channels=interm_dim, out_channels=interm_dim, kernel_size=1) for _ in range(n_layers - 1)
            ])
            self.final_convolutions = nn.ModuleList([
                nn.Conv1d(interm_dim, feature_dim, 3, padding=1)  for i in range(n_layers)
            ])
        
        self.n_layers = n_layers
        self.bottom_up_only = bottom_up_only
        
    def forward(self, x):
        '''
        Expected shape: N, L, 26
        '''
        out = self.conv0(x.transpose(1, 2))
        bottom_ups = [out]
        for layer in self.pyramid_layers:
            out = layer(out)
            bottom_ups.append(out)
        if self.bottom_up_only:
            return bottom_ups
        top_downs = [bottom_ups.pop()]
        for side_layer in self.p_convolutions[::-1]:
            to_add = side_layer(bottom_ups.pop())
            out = self.up(out)[..., :to_add.shape[-1]] + to_add
            top_downs.insert(0, out)
        fpn_out = [final_layer(tens) for (final_layer, tens) in zip(self.final_convolutions, top_downs)]
        return fpn_out


class SelfAttention(nn.Module):

    def __init__(self, input_dim, inner_dim):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Linear(input_dim, inner_dim)
        self.key = nn.Linear(input_dim, inner_dim)
        self.value = nn.Linear(input_dim, inner_dim)
        self.final_projection = nn.Linear(inner_dim, input_dim)


    def forward(self, inpt):
        
        size = inpt.size()
        if len(size) == 4:
            bs, input_dim, height, width = size
            flatten = 2
        elif len(size) == 3:
            bs, input_dim, height = size
            width = 1
            flatten = 1
            
        L = height * width
        x = inpt.flatten(start_dim=-flatten)
        x = x.transpose(1, 2).contiguous().flatten(end_dim=-2)
        
        queries = self.query(x).view(bs, L, -1)
        keys = self.key(x).view(bs, L, -1)
        values = self.value(x).view(bs, L, -1)
        
        factors = torch.softmax(torch.matmul(queries, keys.transpose(1, 2)) / np.round(np.sqrt(queries.size(-1)), 2), dim=-1)
        context_vect = torch.matmul(factors, values)
        context_vect = self.final_projection(context_vect.flatten(end_dim=-2)).view(bs, L, input_dim).transpose(1, 2).contiguous()

        return inpt + context_vect.view(size)