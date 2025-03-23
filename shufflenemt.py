#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2024/1/3 10:14
@File:          shufflenemt.py
'''

from typing import Any, Callable, List, TypeVar, Tuple, Type
from functools import partial
from torch import Tensor
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function

from utils import _log_api_usage_once


M = TypeVar("M", bound=nn.Module)
BUILTIN_MODELS = {}


def register_model(name: str | None = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
    def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
        key = name if name is not None else fn.__name__
        if key in BUILTIN_MODELS:
            raise ValueError(f"An entry is already registered under the name '{key}'.")
        BUILTIN_MODELS[key] = fn
        return fn

    return wrapper


class PFLUFunction(Function):
    '''引自论文 PFLU and FPFLU：Two novel non-monotonic activation functions in convolutional neural networks
    '''

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        sqrt_s_y = x * torch.rsqrt(torch.square(x) + 1)
        y = x * (1 + sqrt_s_y) * 0.5
        return y

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Tensor], output: Tensor) -> None:
        x, = inputs
        sqrt_s_y = x * torch.rsqrt(torch.square(x) + 1)
        ctx.save_for_backward(sqrt_s_y)

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Tensor | None:
        sqrt_s_y, = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = (1 + (2 - torch.square(sqrt_s_y)) * sqrt_s_y) * 0.5
            grad_x = grad_output * grad_x
        return grad_x


class PFLU(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, inputs: Tensor) -> Tensor:
        return PFLUFunction.apply(inputs)

    def extra_repr(self) -> str:
        return f"inplace = {self.inplace}"


def shuffle(inputs: Tensor, chunks: int, axis: int) -> Tensor:
    if axis < 0:
        axis = inputs.dim() + axis

    xs = torch.chunk(inputs, chunks, dim=axis)
    x = torch.stack(xs, dim=axis + 1)
    x = x.flatten(start_dim=axis, end_dim=axis + 1)

    return x


class ViMultiHeadAttentionBase_(nn.Module):
    def __init__(self, in_size: int | Tuple[int, int, int],
                 head_size: int,
                 heads: int = 1,
                 key_size: int | None = None,
                 out_size: int | None = None,
                 attn_dropout: float = 0.,
                 proj_dropout: float = 0.,
                 attn_scale: float | None = 1.,
                 return_attn_scores: bool = True,
                 linear_layer: Callable[..., nn.Module] | None = None) -> None:
        key_size = key_size or head_size
        out_size = out_size or heads * head_size
        attn_scale = attn_scale or key_size ** -0.5
        if linear_layer is None:
            linear_layer = partial(nn.Linear, bias=False)
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.heads = heads
        self.key_size = key_size
        self.out_size = out_size
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout
        self.attn_scale = attn_scale
        self.return_attn_scores = return_attn_scores

        self.o_linear = linear_layer(heads * head_size, out_size)

    def forward(self, inputs: Tensor | Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor | None]:
        raise NotImplementedError

    def pay_attention_to(self, inputs: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """实现标准的乘性多头注意力
        Q: shape = (batch_size, heads, q_H * q_W, key_size)
        K: shape = (batch_size, heads, H * W, key_size)
        V: shape = (batch_size, heads, H * W, head_size)
        attn_dropout_rate:
        说明: 这里单独分离出pay_attention_to函数，是为了方便继承此类来定义不同形式的attention；
             此处要求返回o.shape = (..., heads, q_H * q_W, head_size)。
        """

        qw, kw, vw = inputs
        attn_scores = torch.einsum("bhid, bhjd->bhij", qw, kw)
        attn_scores = attn_scores * self.attn_scale
        a = torch.softmax(attn_scores, -1)
        a = F.dropout(a, p=self.attn_dropout, training=self.training)
        o = torch.einsum("bhij, bhjd->bhid", a, vw)
        return o, attn_scores


class ViMultiHeadSelfAttention(ViMultiHeadAttentionBase_):
    def __init__(self, in_size: int,
                 head_size: int,
                 heads: int = 1,
                 key_size: int | None = None,
                 out_size: int | None = None,
                 attn_dropout: float = 0.,
                 proj_dropout: float = 0.,
                 attn_scale: float | None = None,
                 return_attn_scores: bool = True,
                 linear_layer: Callable[..., nn.Module] | None = None) -> None:
        if linear_layer is None:
            linear_layer = partial(nn.Linear, bias=False)
        super().__init__(in_size,
                         head_size,
                         heads=heads,
                         key_size=key_size,
                         out_size=out_size,
                         attn_dropout=attn_dropout,
                         proj_dropout=proj_dropout,
                         attn_scale=attn_scale,
                         return_attn_scores=return_attn_scores,
                         linear_layer=linear_layer)

        self.i_linear = nn.Linear(in_size, heads * (2 * self.key_size + head_size))

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor] | Tensor:
        B, H, W, _ = inputs.size()

        # 仿射变换
        # qw, kw, vw = self.q_linear(inputs), self.k_linear(inputs), self.v_linear(inputs)
        base = self.i_linear(inputs)
        qw = base[..., :self.heads * self.key_size]
        kw = base[..., self.heads * self.key_size:2 * self.heads * self.key_size]
        vw = base[..., 2 * self.heads * self.key_size:]

        # 形状变换
        qw = qw.reshape(B, H * W, self.heads, self.key_size).transpose(1, 2)
        kw = kw.reshape(B, H * W, self.heads, self.key_size).transpose(1, 2)
        vw = vw.reshape(B, H * W, self.heads, self.head_size).transpose(1, 2)

        # Attention
        o, attn_scores = self.pay_attention_to((qw, kw, vw))

        # 形状逆变换
        o = o.transpose(1, 2)
        o = torch.flatten(o, start_dim=-2).reshape(B, H, W, self.heads * self.head_size)

        # 完成输出
        o = self.o_linear(o)
        o = F.dropout(o, p=self.proj_dropout, training=self.training)
        # 返回结果
        if self.return_attn_scores:
            return o, attn_scores
        else:
            return o


class AttentionShuffleBlock(nn.Module):
    def __init__(self, units: int,
                 heads: int = 2,
                 scale_path_rate: float = 1.,
                 linear_layer: Callable[..., nn.Module] | None = None) -> None:
        assert units % 2 == 0
        if linear_layer is None:
            linear_layer = partial(nn.Linear, bias=False)
        branch_units = units // 2
        super().__init__()

        self.scale_path_rate = scale_path_rate
        self.pre_norm = nn.LayerNorm(branch_units)
        self.attn = ViMultiHeadSelfAttention(branch_units,
                                             branch_units,
                                             heads=heads,
                                             out_size=branch_units,
                                             attn_scale=None,
                                             return_attn_scores=False,
                                             linear_layer=linear_layer)

    def forward(self, inputs: Tensor) -> Tensor:
        i, x = torch.chunk(inputs, 2, dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.scale_path_rate * self.attn(self.pre_norm(x))
        x = x.permute(0, 3, 1, 2)
        o = torch.cat((i, x), 1)
        o = shuffle(o, 2, 1)
        return o


class ConvBlock(nn.Sequential):
    def __init__(self, units: int,
                 kernel_size: List[int] | Tuple[int, int],
                 conv_layer: Callable[..., nn.Module],
                 norm_layer: Callable[..., nn.Module],
                 activation_layer: Callable[..., nn.Module]) -> None:

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        layers = [
            conv_layer(units, units, 1),
            norm_layer(units),
            activation_layer(),
            conv_layer(units, units, kernel_size, padding=padding, groups=units),
            norm_layer(units),
            activation_layer(),
            conv_layer(units, units, 1)
        ]
        super().__init__(*layers)


class ConvShuffleBlock(nn.Module):
    __N = 2

    def __init__(self, units: int,
                 kernel_size: int | Tuple[int, int] | List[int] = 3,
                 scale_path_rate: float = 1.,
                 conv_layer: Callable[..., nn.Module] | None = None,
                 norm_layer: Callable[..., nn.Module] | None = None,
                 activation_layer: Callable[..., nn.Module] | None = None) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.__N
        if conv_layer is None:
            conv_layer = partial(nn.Conv2d, bias=False)
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, momentum=0.01)
        if activation_layer is None:
            activation_layer = PFLU
        super().__init__()

        self.scale_path_rate = scale_path_rate

        branch_units = units // 2

        self.pre_norm = norm_layer(branch_units)

        self.conv_block = ConvBlock(
            branch_units,
            kernel_size,
            conv_layer,
            norm_layer,
            activation_layer
        )

    def forward(self, inputs: Tensor) -> Tensor:
        i, x = torch.chunk(inputs, 2, dim=1)
        x = self.scale_path_rate * self.conv_block(self.pre_norm(x))
        out = torch.cat((i, x), 1)
        out = shuffle(out, 2, 1)
        return out


class ConvResidualBlock(nn.Module):
    __N = 2

    def __init__(self, units: int,
                 kernel_size: int | Tuple[int, int] = 3,
                 scale_path_rate: float = 1.,
                 conv_layer: Callable[..., nn.Module] | None = None,
                 norm_layer: Callable[..., nn.Module] | None = None,
                 activation_layer: Callable[..., nn.Module] | None = None) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.__N
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        if conv_layer is None:
            conv_layer = partial(nn.Conv2d, bias=False)
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, momentum=0.01)
        if activation_layer is None:
            activation_layer = PFLU
        super().__init__()

        self.scale_path_rate = scale_path_rate

        self.pre_norm = norm_layer(units)
        self.conv_block = nn.Sequential(
            conv_layer(units, units, kernel_size, padding=padding),
            norm_layer(units),
            activation_layer(),
            conv_layer(units, units, 1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.scale_path_rate * self.conv_block(self.pre_norm(inputs))


class StackDownsampling(nn.Sequential):
    __N = 2

    def __init__(self, in_units: int,
                 out_units: int,
                 kernel_size: int | Tuple[int, int] = 3,
                 stride: int | Tuple[int, int] = 2,
                 conv_layer: Callable[..., nn.Module] | None = None,
                 norm_layer: Callable[..., nn.Module] | None = None,
                 activation_layer: Callable[..., nn.Module] | None = None) -> None:
        assert in_units != out_units
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.__N
        if isinstance(stride, int):
            stride = (stride,) * self.__N
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        if conv_layer is None:
            conv_layer = partial(nn.Conv2d, bias=False)
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, momentum=0.01)
        if activation_layer is None:
            activation_layer = PFLU

        layers = [
            conv_layer(in_units, out_units // 2, kernel_size, padding=padding, stride=stride),
            norm_layer(out_units // 2),
            activation_layer(),
            conv_layer(out_units // 2, out_units, kernel_size, padding=padding, stride=stride),
        ]

        super().__init__(*layers)


class Downsampling(nn.Sequential):
    __N = 2

    def __init__(self, in_units: int,
                 out_units: int,
                 kernel_size: int | Tuple[int, int] = 3,
                 stride: int | Tuple[int, int] = 2,
                 conv_layer: Callable[..., nn.Module] | None = None,
                 norm_layer: Callable[..., nn.Module] | None = None,
                 activation_layer: Callable[..., nn.Module] | None = None) -> None:
        assert in_units != out_units
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.__N
        if isinstance(stride, int):
            stride = (stride,) * self.__N
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        if conv_layer is None:
            conv_layer = partial(nn.Conv2d, bias=False)
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, momentum=0.01)
        if activation_layer is None:
            activation_layer = PFLU

        hidden_units = out_units // 2

        layers = [
            norm_layer(in_units),
            conv_layer(in_units, hidden_units, 1),
            norm_layer(hidden_units),
            activation_layer(),
            conv_layer(hidden_units, hidden_units, kernel_size, padding=padding, stride=stride, groups=hidden_units),
            norm_layer(hidden_units),
            activation_layer(),
            conv_layer(hidden_units, out_units, 1)
        ]

        super().__init__(*layers)


class ShuffleNeMt(nn.Module):
    def __init__(self, stages_repeats: List[int],
                 stages_out_units: List[int],
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 conv_layer: Callable[..., nn.Module] | None = None,
                 norm_layer: Callable[..., nn.Module] | None = None,
                 activation_layer: Callable[..., nn.Module] | None = None,
                 linear_layer: Callable[..., nn.Module] | None = None,
                 **kwargs) -> None:

        assert len(stages_repeats) == 4, "expected stages_repeats as list of 4 positive ints"
        assert len(stages_out_units) == 5, "expected stages_out_units as list of 5 positive ints"

        if conv_layer is None:
            conv_layer = partial(nn.Conv2d, bias=False)
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, momentum=0.01)
        if activation_layer is None:
            activation_layer = PFLU
        if linear_layer is None:
            linear_layer = partial(nn.Linear, bias=False)

        super().__init__()
        _log_api_usage_once(self)

        self._factory_kwargs = {"conv_layer": conv_layer, "norm_layer": norm_layer, "activation_layer": activation_layer}
        self._linear_layer = linear_layer

        self.total_depth = sum(stages_repeats) - 4 + 1   # 减去4是为了排除4层下采样层，加1是为了视觉自注意力层
        self.scale_path_rates = [1 / math.sqrt(N) for N in range(1, self.total_depth + 1)][::-1]
        self.cur_block_idx = 0
        self.in_units = 3

        self.layer1 = self._make_layers(stages_out_units[0], stages_repeats[0],
                                        skip_connection=ConvResidualBlock,
                                        downsampling_layer=partial(StackDownsampling, stride=1))
        self.layer2 = self._make_layers(stages_out_units[1], stages_repeats[1],
                                        skip_connection=ConvShuffleBlock,
                                        downsampling_layer=partial(Downsampling, stride=2))
        self.layer3 = self._make_layers(stages_out_units[2], stages_repeats[2],
                                        skip_connection=ConvShuffleBlock,
                                        downsampling_layer=partial(Downsampling, stride=2))
        self.layer4 = self._make_layers(stages_out_units[3], stages_repeats[3],
                                        skip_connection=ConvShuffleBlock,
                                        downsampling_layer=partial(Downsampling, stride=2))

        self.conv5 = nn.Sequential(
            norm_layer(self.in_units),
            activation_layer(),
            conv_layer(self.in_units, stages_out_units[-1], 1),
            activation_layer(),
            nn.Dropout(p=dropout_rate, inplace=True)
        )

        self.fc = nn.Linear(stages_out_units[-1], num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layers(self, units: int,
                     repeats_per_stage: int,
                     skip_connection: Type[ConvResidualBlock | ConvShuffleBlock] = ConvShuffleBlock,
                     downsampling_layer: Type[StackDownsampling | Downsampling] = Downsampling) -> nn.Sequential:
        layers: List[nn.Module] = [downsampling_layer(self.in_units,
                                                      units,
                                                      **self._factory_kwargs)]
        self.in_units = units

        for i in range(repeats_per_stage - 1):
            layers.append(skip_connection(units,
                                          scale_path_rate=self.scale_path_rates[self.cur_block_idx],
                                          **self._factory_kwargs))
            self.cur_block_idx += 1
            if self.cur_block_idx == self.total_depth // 2:
                layers.append(AttentionShuffleBlock(units,
                                                    heads=1,
                                                    scale_path_rate=self.scale_path_rates[self.cur_block_idx],
                                                    linear_layer=self._linear_layer))
                self.cur_block_idx += 1
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = torch.mean(x, (-2, -1))
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _shufflenemt(*args: Any, **kwargs: Any) -> ShuffleNeMt:
    model = ShuffleNeMt(*args, **kwargs)
    return model


@register_model()
def shufflenemt_x0_5(**kwargs: Any) -> ShuffleNeMt:
    return _shufflenemt([3, 4, 6, 4], [24, 48, 96, 192, 1024], **kwargs)


@register_model()
def shufflenemt_x1_0(**kwargs: Any) -> ShuffleNeMt:
    return _shufflenemt([3, 4, 6, 4], [24, 116, 232, 464, 1024], **kwargs)


@register_model()
def shufflenemt_x1_5(**kwargs: Any) -> ShuffleNeMt:
    return _shufflenemt([3, 4, 6, 4], [24, 176, 352, 704, 1024], **kwargs)


@register_model()
def shufflenemt_x2_0(**kwargs: Any) -> ShuffleNeMt:
    return _shufflenemt([3, 4, 6, 4], [24, 244, 488, 976, 2048], **kwargs)
