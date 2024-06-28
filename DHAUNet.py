import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
# from pdb import set_trace as stx
from torch.nn import init
import numpy as np
from einops import rearrange
import numbers
from timm.models.layers import to_2tuple, trunc_normal_
import math
from thop import profile
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


##########################################################################



##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, in_c):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, in_c, kernel_size, bias=bias)
        self.conv3 = conv(in_c, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.channels = out_size
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(in_size, out_size, 3, 1, 1)
            self.conv_q = nn.Sequential(
                nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels,
                          bias=True)
            )
            self.conv_kv = nn.Sequential(
                nn.Conv2d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=1, stride=1,
                          bias=True),
                nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1,
                          groups=self.channels * 2, bias=True)
            )
            self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1,
                                      bias=True)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        out1 = out
        if enc is not None and dec is not None:
            assert self.use_csff
            #             print('enc+dec:',out.shape,enc.shape,dec.shape)
            encdec = self.csff_enc(enc) + self.csff_dec(dec)
            b, c, h, w = encdec.shape
            enc_ln = self.norm1(encdec)
            out_ln = self.norm2(out1)
            q = self.conv_q(out_ln)
            q = q.view(b, c, -1)
            k, v = self.conv_kv(enc_ln).chunk(2, dim=1)
            k = k.view(b, c, -1)
            v = v.view(b, c, -1)
            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)
            att = torch.matmul(q, k.permute(0, 2, 1))
            att = self.softmax(att)
            out2 = torch.matmul(att, v).view(b, c, h, w)
            out1 = self.conv_out(out2) + out1

            # out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out1)
            return out_down, out1
        else:
            return out1
class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(out_size * 2, out_size, False, relu_slope)

    def forward(self, x, bridge,):
        #         print('merge',x.shape,bridge.shape)
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff, depth=4):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()  # []
        self.depth = depth
        for i in range(depth - 1):
            #             downsample = True if (i+1) < depth else False
            self.body.append(
                UNetConvBlock(in_size=n_feat + scale_unetfeats * i, out_size=n_feat + scale_unetfeats * (i + 1),
                              downsample=True, relu_slope=0.2, use_csff=True, use_HIN=True))

        self.body.append(UNetConvBlock(in_size=n_feat + scale_unetfeats * (depth - 1),
                                       out_size=n_feat + scale_unetfeats * (depth - 1), downsample=False,
                                       relu_slope=0.2, use_csff=True, use_HIN=True))

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res = []
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                #                 print(encoder_outs[i].shape,decoder_outs[-i-1].shape)
                #                 exit(0)
                if (i + 1) < self.depth:
                    x, x_up = down(x, encoder_outs[i], decoder_outs[-i - 1])
                    res.append(x_up)
                else:
                    #                     print(i,len(encoder_outs),len(decoder_outs))
                    x = down(x)  # ,encoder_outs[i],decoder_outs[-i-1])
        else:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    x, x_up = down(x)
                    res.append(x_up)
                else:
                    x = down(x)
        return res, x


class Decoder_1(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4):
        super(Decoder_1, self).__init__()
        self.up = nn.ConvTranspose2d(n_feat + scale_unetfeats * (depth -1), n_feat + scale_unetfeats * (depth - 2), kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock((n_feat + scale_unetfeats * (depth - 2)) * 2, n_feat + scale_unetfeats * (depth - 2), False, relu_slope=0.2)
        self.skip_conv = (nn.Conv2d(n_feat + scale_unetfeats * (depth - 1), n_feat + scale_unetfeats * (depth - 2), 3, 1, 1))
        self.skip_conv1 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat + scale_unetfeats * (depth - 2), 1, 1, 0),
            nn.GELU())

    def forward(self, x, bridges, x_g3):
        #         for b in bridges:
        #             print(b.shape)

        bridge1 = self.skip_conv(bridges[-1])+self.skip_conv1(x_g3)
        up = self.up(x)
        out = torch.cat([up, bridge1], 1)
        out = self.conv_block(out)
        return out

class Decoder_2(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4):
        super(Decoder_2, self).__init__()
        self.up = nn.ConvTranspose2d(n_feat + scale_unetfeats * (depth - 2), n_feat + scale_unetfeats * (depth - 3), kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock((n_feat + scale_unetfeats * (depth - 3)) * 2, n_feat + scale_unetfeats * (depth - 3), False, relu_slope=0.2)
        self.skip_conv = (nn.Conv2d(n_feat + scale_unetfeats * (depth -2), n_feat + scale_unetfeats * (depth - 3), 3, 1,
                          1))
        self.skip_conv1 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat + scale_unetfeats * (depth - 3), 1, 1, 0),
            nn.Tanh())

    def forward(self, x, bridges,x_g2):
        #         for b in bridges:
        #             print(b.shape)
        bridge2 = self.skip_conv(bridges[-2])+self.skip_conv1(x_g2)
        up = self.up(x)
        out = torch.cat([up, bridge2], 1)
        out = self.conv_block(out)
        return out
class Decoder_3(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4):
        super(Decoder_3, self).__init__()
        self.up = nn.ConvTranspose2d(n_feat + scale_unetfeats * (depth - 3), n_feat + scale_unetfeats * (depth - 4), kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock((n_feat + scale_unetfeats * (depth - 4)) * 2, n_feat + scale_unetfeats * (depth - 4), False, relu_slope=0.2)
        self.skip_conv = (nn.Conv2d(n_feat + scale_unetfeats * (depth -3), n_feat + scale_unetfeats * (depth - 4), 3, 1,
                          1))
        self.skip_conv1 = nn.Sequential(nn.Conv2d(n_feat, n_feat + scale_unetfeats * (depth - 4), 1, 1, 0))

    def forward(self, x, bridges,x_g1):
        #         for b in bridges:
        #             print(b.shape)
        bridge3 = self.skip_conv(bridges[-3])+self.skip_conv1(x_g1)
        up = self.up(x)
        out = torch.cat([up, bridge3], 1)
        out = self.conv_block(out)
        return out


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats,  depth=3):
        super(Decoder, self).__init__()
        self.decoder_1 = Decoder_1(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,)
        self.decoder_2 = Decoder_2(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,)
        self.decoder_3 = Decoder_3(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,)

    def forward(self, out, bridges, x_g1, x_g2, x_g3):
        res = []
        out1 = self.decoder_1(out, bridges, x_g3)
        res.append(out1)
        out2 = self.decoder_2(out1, bridges, x_g2)
        res.append(out2)
        out3 = self.decoder_3(out2, bridges, x_g1)
        res.append(out3)
        return res


##########################################################################
##---------- Resizing Modules ----------



##########################################################################
##Channel and Spatial Attention

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        return x
    def flops(self, ):
        flops = 0
        # fc1
        H,W=32,32
        flops += H*W*80*212
        # dwconv
        flops += H*W*414*3*3
        # fc2
        flops += H*W*212*80
        print("LeFF:{%.2f}"%(flops/1e9))
        return flops


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        #  relative position encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, q_num, kv_num):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        # N = self.win_size[0]*self.win_size[1]
        # nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))

        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num

        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
        print("MCA:{%.2f}" % (flops / 1e9))
        return flops


## Spatial-wise window-based Transformer block (STB in this paper)
class SpatialTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.01,  norm_layer=nn.LayerNorm,bias=False):
        super(SpatialTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FFN(dim, mlp_ratio, bias)

    def forward(self, x):
        b, c, h, w = x.shape

        x = to_3d(x)
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        # padding
        size_par = self.window_size
        pad_l = pad_t = 0
        pad_r = (size_par - w % size_par) % size_par
        pad_b = (size_par - h % size_par) % size_par
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape
        x_size = (Hd, Wd)
        shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, Hd, Wd)  # b h' w' c
        x = shifted_x
        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = to_4d(x, h, w)

        return x

    def flops(self):
        flops = 0
        H, W = 32,32


        flops += self.dim * H * W
        flops += self.attn.flops(H * W, 8 * 8)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops( )
        print("LeWin:{%.2f}"%(flops/1e9))
        return flops


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


## Channel-wise cross-covariance Transformer block (CTB in this paper)
class ChannelTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(ChannelTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class Downsample1(nn.Module):
    def __init__(self, n_feat):
        super(Downsample1, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)
class Expend_head(nn.Module):
    def __init__(self, in_size, out_size,  relu_slope,  ):
        super(Expend_head, self).__init__()
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)
        return out

##########################################################################
class ExpendNet(nn.Module):
    def __init__(self, n_feat=32, out_nc=5, c=64, lightweight=False):
        super(ExpendNet, self).__init__()
        self.expanding_block = nn.Sequential(
            ChannelTransformerBlock(dim=n_feat, num_heads=1, ffn_expansion_factor=2.66, bias=False,
                                    LayerNorm_type=WithBias_LayerNorm),
            ChannelTransformerBlock(dim=n_feat, num_heads=2, ffn_expansion_factor=2.66, bias=False,
                                    LayerNorm_type=WithBias_LayerNorm))
        self.detail_head = Expend_head(in_size=2*n_feat, out_size=n_feat, relu_slope=0.2,)
        self.structure_head = Expend_head(in_size=2*n_feat, out_size=n_feat, relu_slope=0.2, )
        self.down1_1 = Downsample1(n_feat)
        self.down1_2 = Downsample1(n_feat)
        self.beta = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = x
        x1 = self.expanding_block(x)
        x_1 = res + x1 * self.beta
        x_1_1 = self.down1_1(x_1)
        x_2 = self.structure_head(x_1_1)
        x_2_1 = self.down1_2(x_2)
        x_3 = self.detail_head(x_2_1)
        return x_1, x_2, x_3



class HyPaNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=7, c=64):
        super(HyPaNet, self).__init__()
        self.out_nc = out_nc
        self.feature1 = nn.Conv2d(in_channels=in_nc, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
                                  bias=True)
        self.feature2 = nn.Conv2d(in_channels=c, out_channels=out_nc, kernel_size=3, padding=1, stride=1, groups=1,
                                  bias=True)
        self.conv_1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True))
        self.conv11 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.feature1(x)
        x = self.conv_1(x)
        x = x * self.sca(x)
        x = self.feature2(x)
        y = self.softplus(x)
        return y


class PP(nn.Module):
    def __init__(self, ):
        super(PP, self).__init__()
        self.cons = nn.Parameter(torch.Tensor([0.6]).repeat(7))
        self.stages = 6
        self.cons1 = nn.Parameter(torch.Tensor([0.4]).repeat(7))

        self.stage_estimator = HyPaNet(in_nc=1, out_nc=1, c=64)

    def forward(self, mu_all, x):
        # [b, stages, h, w]
        for i in range(self.stages):
            delta_mu = self.stage_estimator(x)
            mu = self.cons1[i]*mu_all[:, i:i + 1, :, :] + self.cons[i] * delta_mu
            return mu + 1


##########################################################################
class MPRBlock(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=80, scale_unetfeats=48, kernel_size=3,
                 reduction=4, bias=False, block_size=32,depth=4):
        super(MPRBlock, self).__init__()
        act = nn.PReLU()
        self.block_size = block_size
        self.shallow_feat = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))

        # Cross Stage Feature Fusion (CSFF)
        self.stage_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.sam = SAM(n_feat, kernel_size=1, bias=bias, in_c=in_c)
        self.P = PP()
        self.concat = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.expanding_block = ExpendNet(n_feat)
        self.SpatialTransformerBlock = nn.Sequential(
            SpatialTransformerBlock(dim=n_feat + scale_unetfeats * (depth - 1), num_heads=1, window_size=8,
                                    mlp_ratio=2.66, drop_path=0.01),
            SpatialTransformerBlock(dim=n_feat + scale_unetfeats * (depth - 1), num_heads=2, window_size=8,
                                    mlp_ratio=2.66, drop_path=0.01))

    def forward(self, stage_img, mu_all, x_samfeats, f_encoder, f_decoder, PhiTPhi, PhiTb):
        b, c, w, h = stage_img.shape
        x_k_1 = F.unfold(stage_img, kernel_size=self.block_size, stride=self.block_size).permute(0, 2,
                                                                                                 1).contiguous()  # (b,l,cwh)
        l = x_k_1.shape[1]
        x_k_1 = x_k_1.view(b * l, -1)
        P = F.unfold(self.P(mu_all, stage_img), kernel_size=self.block_size, stride=self.block_size).permute(0, 2,
                                                                                                             1).contiguous()  # (b,l,cwh)
        P = P.view(b * l, -1)
        x = x_k_1 - torch.div(torch.mm(x_k_1, PhiTPhi), P)
        r_k = x + torch.div(PhiTb, P)

        # x = x_k_1 - self.r * torch.mm(x_k_1, PhiTPhi)
        # r_k = x + self.r * PhiTb

        r_k = r_k.view(b, l, -1).permute(0, 2, 1).contiguous()  # view(b,l,w,h)
        r_k = F.fold(r_k, output_size=(w, h), kernel_size=self.block_size, stride=self.block_size).contiguous()

        # compute x_k
        x = self.shallow_feat(r_k)
        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x_cat = self.concat(torch.cat([x, x_samfeats], 1))
        x_g1, x_g2, x_g3 = self.expanding_block(x_cat)
        ## Process features of both patches with Encoder of Stage 2
        feat1, f_encoder = self.stage_encoder(x_cat, f_encoder, f_decoder)
        f_encoder = self.SpatialTransformerBlock(f_encoder)

        ## Pass features through Decoder of Stage 2
        f_decoder = self.stage_decoder(f_encoder, feat1,x_g1, x_g2, x_g3)
        ## Apply SAM
        x_samfeats, stage_img = self.sam(f_decoder[-1], r_k)
        return stage_img, x_samfeats, feat1, f_decoder


##########################################################################


class DHAUNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=32, scale_unetfeats=16, scale_orsnetfeats=16, num_cab=8, kernel_size=3,
                 reduction=4, bias=False, nums_stages=4, cs_ratio=25, block_size=32, depth=4):
        super(DHAUNet, self).__init__()
        self.n_input = int(cs_ratio * 1024)
        self.block_size = block_size

        self.Phi = nn.Parameter(
            init.xavier_normal_(torch.Tensor(np.ceil(cs_ratio * 0.01 * 1024.).astype(int), 1024)))
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
        self.body = nn.ModuleList()
        for _ in range(nums_stages):
            self.body.append(MPRBlock(
                in_c=in_c, out_c=in_c, n_feat=n_feat, scale_unetfeats=scale_unetfeats, kernel_size=kernel_size,
                reduction=reduction, bias=bias
            ))
        self.shallow_feat_final = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        #         self.stage_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats,
        #                                     num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias, in_c=in_c)
        self.global_prior = HyPaNet(in_nc=1, out_nc=7, c=64)
        self.P = PP()
        # self.attention = step(in_c=16)
        self.concat_final = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)
        self.expanding_block = ExpendNet(n_feat)
        self.SpatialTransformerBlock = nn.Sequential(
            SpatialTransformerBlock(dim=n_feat + scale_unetfeats * (depth - 1), num_heads=1, window_size=8,
                                    mlp_ratio=2.66, drop_path=0.01),
            SpatialTransformerBlock(dim=n_feat + scale_unetfeats * (depth - 1), num_heads=2, window_size=8,
                                    mlp_ratio=2.66, drop_path=0.01))

    def forward(self, img):
        output_ = []
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        b, c, w, h = img.shape

        blocks = F.unfold(img, kernel_size=self.block_size, stride=self.block_size).permute(0, 2,
                                                                                            1).contiguous()  # (b,l,cwh)
        l = blocks.shape[1]
        blocks = blocks.view(b * l, -1)
        PhiTPhi = torch.mm(torch.transpose(self.Phi, 0, 1), self.Phi)  # torch.mm(Phix, Phi)
        Phix = torch.mm(blocks, torch.transpose(self.Phi, 0, 1))  # compression result
        PhiTb = torch.mm(Phix, self.Phi)  # .view(b,l,-1).permute(0, 2, 1).contiguous()
        y_0 = PhiTb.view(b, l, -1).permute(0, 2, 1).contiguous()  # view(b,l,w,h)
        y_0 = F.fold(y_0, output_size=(w, h), kernel_size=self.block_size, stride=self.block_size).contiguous()
        mu_all = self.global_prior(y_0)
        P_0 = F.unfold(self.P(mu_all, y_0), kernel_size=self.block_size, stride=self.block_size).permute(0, 2,
                                                                                                         1).contiguous()  # (b,l,cwh)
        P_0 = P_0.view(b * l, -1)
        # compute r_0
        x_0 = PhiTb  # .view(b,-1)
        #         print(x_0.shape, PhiTPhi.shape)
        #         exit(0)
        #         r0= self.attention(img,self.Phi)
        #         x = x_0 - self.r0 * torch.mm(x_0, PhiTPhi)
        #         r_0 = x + self.r0 * PhiTb
        x = x_0 - torch.div(torch.mm(x_0, PhiTPhi), P_0)
        r_0 = x + torch.div(PhiTb, P_0)
        r_0 = r_0.view(b, l, -1).permute(0, 2, 1).contiguous()  # view(b,l,w,h)
        r_0 = F.fold(r_0, output_size=(w, h), kernel_size=self.block_size, stride=self.block_size).contiguous()

        # compute x_k
        x = self.shallow_feat1(r_0)
        x_g1,x_g2,x_g3 = self.expanding_block(x)
        ## Process features of all 4 patches with Encoder of Stage 1
        feat1, f_encoder = self.stage1_encoder(x)
        f_encoder = self.SpatialTransformerBlock(f_encoder)
        ## Pass features through Decoder of Stage 1
        f_decoder = self.stage1_decoder(f_encoder, feat1, x_g1, x_g2, x_g3)
        ## Apply Supervised Attention Module (SAM)
        x_samfeats, stage_img = self.sam12(f_decoder[-1], r_0)
        output_.append(stage_img)

        ##-------------------------------------------
        ##-------------- Stage 2_k-1---------------------
        ##-------------------------------------------
        for stage_model in self.body:
            stage_img, x_samfeats, feat1, f_decoder = stage_model(stage_img, mu_all, x_samfeats, feat1, f_decoder,
                                                                  PhiTPhi, PhiTb)
            output_.append(stage_img)
        ##-------------------------------------------
        ##-------------- Stage k---------------------
        ##-------------------------------------------
        # compute r_k
        x_k_1 = F.unfold(stage_img, kernel_size=self.block_size, stride=self.block_size).permute(0, 2,
                                                                                                 1).contiguous()  # (b,l,cwh)
        x_k_1 = x_k_1.view(b * l, -1)
        P_final = F.unfold(self.P(mu_all, stage_img), kernel_size=self.block_size, stride=self.block_size).permute(0, 2,
                                                                                                                   1).contiguous()  # (b,l,cwh)
        P_final = P_final.view(b * l, -1)
        x = x_k_1 - torch.div(torch.mm(x_k_1, PhiTPhi), P_final)
        r_k = x + torch.div(PhiTb, P_final)
        r_k = r_k.view(b, l, -1).permute(0, 2, 1).contiguous()  # view(b,l,w,h)
        r_k = F.fold(r_k, output_size=(w, h), kernel_size=self.block_size, stride=self.block_size).contiguous()
        # compute x_k
        x = self.shallow_feat_final(r_k)
        ## Concatenate SAM features
        x_cat = self.concat_final(torch.cat([x, x_samfeats], 1))
        #         x_cat = self.stage_orsnet(x_cat, f_encoder, f_decoder)
        stage_img = self.tail(x_cat) + r_k
        output_.append(stage_img)

        return output_[::-1]  # [stage5_img + x3_img, stage4_img, stage3_img, stage2_img, stage1_img]

