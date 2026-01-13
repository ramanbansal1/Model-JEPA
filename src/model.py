import torch
import torch.nn as nn
import MinkowskiEngine as ME


def conv_block(in_ch, out_ch, kernel=3, stride=1):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=kernel, stride=stride, dimension=3),
        ME.MinkowskiBatchNorm(out_ch),
        ME.MinkowskiELU(),
    )


def up_block(in_ch, out_ch, kernel=2, stride=2):
    return nn.Sequential(
        ME.MinkowskiGenerativeConvolutionTranspose(
            in_ch, out_ch, kernel_size=kernel, stride=stride, dimension=3
        ),
        ME.MinkowskiBatchNorm(out_ch),
        ME.MinkowskiELU(),
        ME.MinkowskiConvolution(out_ch, out_ch, kernel_size=3, dimension=3),
        ME.MinkowskiBatchNorm(out_ch),
        ME.MinkowskiELU(),
    )

class SparseEncoder(nn.Module):
    CHANNELS = [1, 16, 32, 64, 128, 256, 512, 1024]

    def __init__(self):
        super().__init__()
        ch = self.CHANNELS

        self.blocks = nn.ModuleList([
            conv_block(ch[0], ch[1], stride=1),   # s1
            nn.Sequential(                        # s1 -> s2
                conv_block(ch[1], ch[2], kernel=2, stride=2),
                conv_block(ch[2], ch[2])
            ),
            nn.Sequential(                        # s2 -> s4
                conv_block(ch[2], ch[3], kernel=2, stride=2),
                conv_block(ch[3], ch[3])
            ),
            nn.Sequential(                        # s4 -> s8
                conv_block(ch[3], ch[4], kernel=2, stride=2),
                conv_block(ch[4], ch[4])
            ),
            nn.Sequential(                        # s8 -> s16
                conv_block(ch[4], ch[5], kernel=2, stride=2),
                conv_block(ch[5], ch[5])
            ),
            nn.Sequential(                        # s16 -> s32
                conv_block(ch[5], ch[6], kernel=2, stride=2),
                conv_block(ch[6], ch[6])
            ),
            nn.Sequential(                        # s32 -> s64
                conv_block(ch[6], ch[7], kernel=2, stride=2),
                conv_block(ch[7], ch[7])
            ),
        ])

    def forward(self, x):
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats  # [s1, s2, s4, s8, s16, s32, s64]

class SparseDecoder(nn.Module):
    CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self):
        super().__init__()
        ch = self.CHANNELS

        self.up_blocks = nn.ModuleList([
            up_block(ch[6], ch[5]),  # s64 -> s32
            up_block(ch[5], ch[4]),  # s32 -> s16
            up_block(ch[4], ch[3]),  # s16 -> s8
            up_block(ch[3], ch[2]),  # s8  -> s4
            up_block(ch[2], ch[1]),  # s4  -> s2
            up_block(ch[1], ch[0]),  # s2  -> s1
        ])

        self.cls_heads = nn.ModuleList([
            ME.MinkowskiConvolution(ch[5], 1, kernel_size=1, dimension=3),
            ME.MinkowskiConvolution(ch[4], 1, kernel_size=1, dimension=3),
            ME.MinkowskiConvolution(ch[3], 1, kernel_size=1, dimension=3),
            ME.MinkowskiConvolution(ch[2], 1, kernel_size=1, dimension=3),
            ME.MinkowskiConvolution(ch[1], 1, kernel_size=1, dimension=3),
            ME.MinkowskiConvolution(ch[0], 1, kernel_size=1, dimension=3),
        ])

        self.pruning = ME.MinkowskiPruning()


class CompletionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SparseEncoder()
        self.decoder = SparseDecoder()

    def get_target(self, out, target_key):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            key = cm.stride(target_key, out.tensor_stride[0])
            kernel_map = cm.kernel_map(out.coordinate_map_key, key, kernel_size=1)
            for _, curr in kernel_map.items():
                target[curr[0].long()] = 1
        return target

    def forward(self, x, target_key):
        enc_feats = self.encoder(x)

        out_cls, targets = [], []
        dec = enc_feats[-1]

        for i, (up, cls) in enumerate(zip(self.decoder.up_blocks, self.decoder.cls_heads)):
            dec = up(dec)
            skip = enc_feats[-(i + 2)]
            dec = dec + skip

            cls_out = cls(dec)
            keep = (cls_out.F > 0).squeeze()

            target = self.get_target(dec, target_key)
            out_cls.append(cls_out)
            targets.append(target)

            if self.training:
                keep |= target

            dec = self.decoder.pruning(dec, keep)

        return out_cls, targets, dec
