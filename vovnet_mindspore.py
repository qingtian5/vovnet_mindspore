from collections import OrderedDict
import mindspore as ms
from mindspore import nn, ops

from mindvision.engine.class_factory import ClassFactory, ModuleType

ms.set_context(mode=ms.GRAPH_MODE)

__all__ = ["VoVNet"]

#  如果为True，则使用当前批处理数据的平均值和方差值，并跟踪运行平均值和运行方差。
# - 如果为False，则使用指定值的平均值和方差值，不跟踪统计值。
_NORM = True

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE,
    "V-57-eSE": VoVNet57_eSE,
    "V-99-eSE": VoVNet99_eSE,
}


def dw_conv3x3(in_channels, out_channels, module_name, postfix,
               stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}_dw_conv3x3'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   pad_mode='pad',
                   padding=padding,
                   group=out_channels,
                   has_bias=False)),
        ('{}_{}_pw_conv1x1'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=1,
                   stride=1,
                   pad_mode='pad',
                   padding=0,
                   group=1,
                   has_bias=False)),
        ('{}_{}_pw_norm'.format(module_name, postfix), nn.BatchNorm2d(out_channels, use_batch_statistics=_NORM)),
        ('{}_{}_pw_relu'.format(module_name, postfix), nn.ReLU()),
    ]


def conv3x3(
        in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1
):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}_conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode='pad',
                padding=padding,
                group=groups,
                has_bias=False,
            ),
        ),
        (f"{module_name}_{postfix}_norm", nn.BatchNorm2d(out_channels, use_batch_statistics=_NORM)),
        (f"{module_name}_{postfix}_relu", nn.ReLU()),
    ]


def conv1x1(
        in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0
):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}_conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode='pad',
                padding=padding,
                group=groups,
                has_bias=False,
            ),
        ),
        (f"{module_name}_{postfix}_norm", nn.BatchNorm2d(out_channels, use_batch_statistics=_NORM)),
        (f"{module_name}_{postfix}_relu", nn.ReLU()),
    ]


class Hsigmoid(nn.Cell):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace
        self.relu6 = nn.ReLU6()

    def construct(self, x):
        return self.relu6(x + 3.0) / 6.0


# nn.AdaptiveAvgPool2D(1) 如何复现,用ops.AdaptiveAvgPool2D(1)替换
class eSEModule(nn.Cell):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = ops.AdaptiveAvgPool2D(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, pad_mode='pad', padding=0, has_bias=True)
        self.hsigmoid = Hsigmoid()

    def construct(self, x):
        _input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return _input * x


class _OSA_module(nn.Cell):
    def __init__(
            self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False
    ):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.CellList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.SequentialCell(
                OrderedDict(conv1x1(in_channel, stage_ch,
                                    "{}_reduction".format(module_name), "0")))
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(
                    nn.SequentialCell(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(
                    nn.SequentialCell(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i)))
                )
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.SequentialCell(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat"))
        )

        self.ese = eSEModule(concat_ch)
        # self.ese = nn.SequentialCell(OrderedDict([(f'{module_name}',eSEModule(concat_ch))]))

    def construct(self, x):

        identity_feat = x

        output = ()
        output += (x,)

        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output += (x,)

        x = ops.Concat(axis=1)(output)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt

        # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # 用 nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same') 代替


@ClassFactory.register(ModuleType.BACKBONE)
class VoVNet(nn.Cell):
    def __init__(self,
                 stem,
                 stage_conv_ch,
                 stage_out_ch,
                 block_per_stage,
                 layer_per_block,
                 eSE,
                 dw,
                 input_ch,
                 out_features=['stage2', 'stage3', 'stage4', 'stage5'],
                 freeze_at=0,
                 norm_bn=None):

        super(VoVNet, self).__init__()

        global _NORM
        _NORM = norm_bn

        stem_ch = stem
        config_stage_ch = stage_conv_ch
        config_concat_ch = stage_out_ch
        block_per_stage = block_per_stage
        layer_per_block = layer_per_block
        SE = eSE
        depthwise = dw

        self._out_features = out_features

        # Stem module
        conv_type = dw_conv3x3 if depthwise else conv3x3
        stem = conv3x3(input_ch, stem_ch[0], "stem", "1", 2)
        stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1)
        stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2)
        self.stem = nn.SequentialCell((OrderedDict(stem)))

        stem_out_ch = [stem_ch[2]]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]

        self.stage2 = self._make_layer(in_ch_list[0], config_stage_ch[0], config_concat_ch[0],
                                       block_per_stage[0], layer_per_block, 2, SE, depthwise)
        self.stage3 = self._make_layer(in_ch_list[1], config_stage_ch[1], config_concat_ch[1],
                                       block_per_stage[1], layer_per_block, 3, SE, depthwise)
        self.stage4 = self._make_layer(in_ch_list[2], config_stage_ch[2], config_concat_ch[2],
                                       block_per_stage[2], layer_per_block, 4, SE, depthwise)
        self.stage5 = self._make_layer(in_ch_list[3], config_stage_ch[3], config_concat_ch[3],
                                       block_per_stage[3], layer_per_block, 5, SE, depthwise)

        # initialize weights
        self._initialize_weights()

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(freeze_at)

    def _make_layer(self,
                    in_ch, stage_ch,
                    concat_ch,
                    block_per_stage,
                    layer_per_block,
                    stage_num,
                    SE=False,
                    depthwise=False):
        layers = []
        if not stage_num == 2:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'))
        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        layers.append(_OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise))
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            layers.append(
                _OSA_module(
                    concat_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, identity=True, depthwise=depthwise
                )
            )
        return nn.SequentialCell(layers)

    def _freeze_backbone(self, freeze_at):
        if freeze_at <= 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "stage" + str(stage_index + 1))
            for p in m.get_parameters():
                p.requires_grad = False

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeUniform(),
                    cell.weight.shape, cell.weight.dtype
                ))

    def construct(self, x):
        outputs = ()
        x = self.stem(x)
        if 'stem' in self._out_features:
            outputs += (x,)
        x = self.stage2(x)
        if 'stage2' in self._out_features:
            outputs += (x,)
        x = self.stage3(x)
        if 'stage3' in self._out_features:
            outputs += (x,)
        x = self.stage4(x)
        if 'stage4' in self._out_features:
            outputs += (x,)
        x = self.stage5(x)
        if 'stage5' in self._out_features:
            outputs += (x,)

        return outputs
