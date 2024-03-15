import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb, slots):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb, slots):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(4, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        # h = self.group_norm(x)
        h = x
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class CrossAttnBlock(nn.Module):#这个是有问题的，你目前的写法是slot个数变成特征图大小，然后这么多个slot去竞争特征图的每一个像素，然后再进行加权求和。这样slot的个数就没什么用。正确的写法应该是K个slot去竞争特征图的每一个像素，然后再进行加权求和。
    def __init__(self, in_ch, num_slots, slot_size):
        super().__init__()
        
        self.group_norm = nn.GroupNorm(4, in_ch)
        self.proj_q = nn.Conv2d(in_ch, num_slots, 1, stride=1, padding=0)#k,q,v只是把每一个像素和每一个slot的维度变成一样，这样才能够进行相似度计算（竞争）
        self.proj_k = nn.Linear(slot_size, slot_size)
        self.proj_v = nn.Linear(slot_size, slot_size)
        self.proj = nn.Conv2d(num_slots, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x, slots):
        # h = self.group_norm(x)
        h = x
        q = self.proj_q(h) # [B, num_slots, H, W]
        k = self.proj_k(slots) # [B, num_slots, slot_size]
        v = self.proj_v(slots) # [B, num_slots, slot_size]

        B, C, H, W = q.shape # C == num_slots
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        # k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5)) # [B, H * W, slot_size]
        # assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        # v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        v = v.permute(0, 2, 1) # [B, slot_size, num_slots]
        h = torch.bmm(w, v) # [B, H * W, num_slots]
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2) # [B, C, H, W]
        h = self.proj(h)

        return x + h

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, num_slots=-1, slot_size=-1):
        super().__init__()
        self.block1 = nn.Sequential(
            # nn.GroupNorm(4, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            # nn.GroupNorm(4, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
            self.crossattn = CrossAttnBlock(out_ch, num_slots, slot_size)
        else:
            self.attn = None
            self.crossattn = None
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb, slots):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        if self.attn is not None:
            h = self.attn(h)
            h = self.crossattn(h, slots)
        return h


class UNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        T = args.dif_T
        ch = args.dif_ch
        ch_mult = args.dif_ch_mult
        attn = args.dif_attn
        num_res_blocks = args.dif_num_res_blocks
        dropout = args.dif_dropout
        image_size = args.dif_image_size
        
        num_slots = args.se_num_slots
        slot_size = args.se_slot_size

        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(args.ae_embed_dim, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), num_slots=num_slots, slot_size=slot_size))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                image_size //= 2
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True, num_slots=num_slots, slot_size=slot_size),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), num_slots=num_slots, slot_size=slot_size))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
                image_size *= 2
        assert len(chs) == 0

        self.tail = nn.Sequential(
            # nn.GroupNorm(4, now_ch),
            Swish(),
            nn.Conv2d(now_ch, args.ae_embed_dim, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, slots):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, slots)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, slots)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, slots)
        h = self.tail(h)

        assert len(hs) == 0
        return h