import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random
import numpy as np

# unfold_A = [
#     [1,2,17],[2,21,1],[3,21,4],[4,3,4],[5,21,6],[6,5,7],[7,6,8],[8,7,23],
#     [9,21,10],[10,9,11],[11,10,12],[12,11,25],[13,1,14],[14,13,15],[15,14,16],[16,15,16],
#     [17,1,18],[18,17,19],[19,18,20],[20,19,20],[21,21,2],
#     [22,23,22],[23,8,22],[24,25,24],[25,12,24],[1,2,13],[21,21,3],[21,21,5],[21,21,9]
# ]
unfold_A = [
    [9,10,11,12,24,25],[5,6,7,8,22,23],[2,3,4,5,9,21],[1,2,17,18,19,20],[1,2,13,14,15,16]
]
for ske in unfold_A:
    for j in range(len(ske)):
        ske[j] = ske[j] - 1

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.BatchNorm1d(dim)
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         x = x.permute(0,2,1).contiguous()
#         x = self.norm(x)
#         x = x.permute(0, 2, 1).contiguous()
#         return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 16, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            # nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dropout = nn.Dropout(p=dropout)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.dropout(x)

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad,
                              stride=stride),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0,
                      stride=1))


        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # conv_init(self.conv)
        # bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(self.conv(x))
        return x

class unit_vt(nn.Module):
    def __init__(self, in_channels, out_channels, V = 25, kernel_size=9, stride=1):
        super(unit_vt, self).__init__()
        self.V = V
        self.tcn = unit_tcn(in_channels, out_channels, kernel_size, stride)



    def forward(self, x):
        N,L,D = x.shape
        x_t = x[:,self.V+1:,:]
        x_t = x_t.permute(0,2,1).contiguous()  #NDL
        x_t = self.tcn(x_t)
        x_t = x_t.permute(0,2,1).contiguous()
        x[:,self.V+1:,:] = x_t
        return x

# class TF_V(nn.Module):
#     def __init__(self, V, T, channels, dim, stride, depth, kernel_size=9,heads=8, dropout=0.0):
#         super().__init__()
#         pad = int((kernel_size - 1) / 2)
#         dim_head = dim//heads
#         mlp_dim = dim
#         self.embedding = nn.Sequential(
#             nn.Conv2d(channels, dim, kernel_size=(kernel_size, 1), padding=(pad, 0),
#                                     stride=(stride, 1)),
#             nn.BatchNorm2d(dim),
#             nn.GELU(),
#         )
#         # self.norm = nn.Sequential(
#         #     nn.BatchNorm2d(dim),
#         #     nn.LayerNorm(dim),
#         #     nn.GELU(),
#         # )
#         # self.embedding_v = nn.Conv1d(channels * T, dim, 1)
#         self.num_patches = V
#         self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
#         self.transformer = Transformer(dim, depth=depth, heads=heads, dim_head=dim_head, \
#                                             mlp_dim=mlp_dim, dropout=dropout)
#         # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#
#     def forward(self, x):
#         N, C, T, V = x.shape
#         x1 = self.embedding(x)
#         C_new = x1.shape[1]
#         x1 = x1.permute(0,2,3,1).contiguous().view(-1,V,C_new)
#         # x1 = self.norm(x1)
#         # x1 = self.embedding_v(x1).permute(0, 2, 1).contiguous()
#         # cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=N)
#         # x1 = torch.cat((cls_tokens, x1), dim=1)
#         x1 += self.pos_embedding
#         x1 = self.transformer(x1)
#         x1 = x1.view(N,-1,V,C_new).permute(0,3,1,2).contiguous()
#         # x1 = x1[:, 1:, :].mean(dim=1)
#         return x1

# class Block_V(nn.Module):
#     def __init__(self, dim, depth, T, V = 25, heads=8, dropout=0.0):
#         super().__init__()
#         num_patches = T
#         self.mlp = nn.Linear(6*V, dim)
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
#         self.tf = Transformer(dim, depth=depth, heads=heads, dim_head=dim // heads, \
#                             mlp_dim=dim, dropout=dropout)
#         self.fc = nn.Linear(dim, 60)
#
#     def forward(self, x):
#         N, C, T, V = x.shape
#         x = x.permute(0, 2, 1, 3).contiguous()  #NTCV
#         x = x.view(N, T,  C*V)
#         x = self.mlp(x)
#         x += self.pos_embedding
#         x = self.tf(x)
#         x = x.mean(1)
#         x = self.fc(x)
#         return x



class Block_V(nn.Module):
    def __init__(self, dim, depth, T, V = 25, heads=8, dropout=0.0):
        super().__init__()
        dim_head = dim//heads
        mlp_dim = dim
        # self.embedding_t = nn.Linear(channels, dim)
        # self.embedding_v = nn.Linear(channels, dim)
        self.transformer_t = nn.ModuleList([])
        self.transformer_v = nn.ModuleList([])
        self.pos_embedding_T = nn.Parameter(torch.randn(1, T, dim))
        self.pos_embedding_V = nn.Parameter(torch.randn(1, V, dim))
        self.dim_net = nn.ModuleList([])
        self.fc = nn.Linear(dim, 60)

        self.gelu = nn.GELU()
        self.depth = depth
        for _ in range(self.depth):
            self.transformer_t.append(
                Transformer(dim, depth=1, heads=heads, dim_head=dim // heads, \
                            mlp_dim=dim, dropout=dropout))
            self.transformer_v.append(
                Transformer(dim, depth=1, heads=heads, dim_head=dim // heads, \
                            mlp_dim=dim, dropout=dropout))
            # self.dim_net.append(nn.Sequential(
            #     nn.BatchNorm2d(dim * 2),
            #     nn.GELU(),
            #     nn.Conv2d(dim * 2, dim, 1),
            #     nn.BatchNorm2d(dim),
            #     nn.GELU(),
            #     ))

    def forward(self, x):
        N, C, T, V = x.shape
        for i in range(self.depth):
            x_v = x.permute(0, 2, 3, 1).contiguous().view(N * T, V, C)
            x_t = x.permute(0, 3, 2, 1).contiguous().view(N * V, T, C)
            if i == 0:
                x_v += self.pos_embedding_V
                x_t += self.pos_embedding_T
            x_v = self.transformer_v[i](x_v)
            x_t = self.transformer_t[i](x_t)
            x_v = x_v.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()  # NCTV
            x_t = x_t.view(N, V, T, -1).permute(0, 3, 2, 1).contiguous()  # NCTV
            x = x_v + x_t

            # x_v = x.permute(0, 2, 3, 1).contiguous().view(N * T, V, C)
            # if i == 0:
            #     x_v += self.pos_embedding_V
            # x_v = self.transformer_v[i](x_v)
            # x_v = x_v.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()  # NCTV
            # x_t = x_v.permute(0, 3, 2, 1).contiguous().view(N * V, T, C)
            # if i == 0:
            #     x_t += self.pos_embedding_T
            # x_t = self.transformer_t[i](x_t)
            # x_t = x_t.view(N, V, T, -1).permute(0, 3, 2, 1).contiguous()  # NCTV
            # x = x_t
        return x

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 6:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y

class unit_TF_V(nn.Module):
    def __init__(self, A, in_channels, out_channels, depth, kernel_size=9, V = 25, heads=8, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_blocks = nn.ModuleList([])
        # self.out_blocks = nn.ModuleList([])
        self.out_cat = nn.ModuleList([])
        for i in range(len(in_channels)):
            self.in_blocks.append(Block_V(in_channels[i], depth, T=self.get_T(in_channels[i]), V=V))
        for i in range(len(self.out_channels)):
            # self.out_blocks.append(Block_V(out_channels[i], depth, T=self.get_T(), V=V))
            # self.out_blocks.append(unit_gcn(out_channels[i], out_channels[i], A, adaptive=True))
            self.layers.append(self.exchange_layer(out_channels[i], i, kernel_size))
            self.out_cat.append(
                nn.Sequential(
                    nn.Conv2d(out_channels[i]*len(self.in_channels), out_channels[i], 1),
                    nn.BatchNorm2d(out_channels[i]),
                    nn.GELU(),
                )
            )
    def get_T(self,c):
        T = [64,32,16,8,4]
        C = [64,128,256,512]
        if c == 3 or c == 6:
            t = 64
        else:
            p = C.index(c)
            t = T[p]
        return t

    def exchange_layer(self, out_channel, idx, kernel_size):
        layers = nn.ModuleList([])
        pad = int((kernel_size - 1) / 2)
        for i in range(len(self.in_channels)):
            if i < idx:
                exchange_layers = nn.Sequential(
                    nn.Conv2d(self.in_channels[i], out_channel, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(2**(idx-i), 1)),
                    nn.BatchNorm2d(out_channel),
                    nn.GELU(),
                )
            elif i == idx:
                exchange_layers = nn.Sequential(
                    nn.Conv2d(self.in_channels[i], out_channel, kernel_size=1),
                    nn.BatchNorm2d(out_channel),
                    nn.GELU(),
                )
            elif i > idx:
                exchange_layers = nn.Sequential(
                    nn.Conv2d(self.in_channels[i], out_channel, kernel_size=1),
                    nn.Upsample(scale_factor=(2**(i-idx), 1), mode='bilinear', align_corners=True),
                    nn.BatchNorm2d(out_channel),
                    nn.GELU(),
                )
            layers.append(exchange_layers)
        return layers
    def forward(self, x_list):
        out_list = []
        for j in range(len(self.in_channels)):
            x_list[j] = self.in_blocks[j](x_list[j])
        for i in range(len(self.out_channels)):
            out = []
            for j in range(len(self.in_channels)):
                out.append(self.layers[i][j](x_list[j]))
            out_list.append(self.out_cat[i](torch.cat(out,dim = 1)))
        return out_list


class ViT(nn.Module):
    def __init__(self, *, A, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        V = 25
        T = 64
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 64, 1),
        )
        self.TF1 = unit_TF_V(A, in_channels=[64],out_channels=[64,128],depth=3,kernel_size=9, V = 25)
        # self.TF2 = unit_TF_V(A, in_channels=[64, 128], out_channels=[64, 128], depth=1, kernel_size=9, V=25)
        # self.TF3 = unit_TF_V(A,in_channels=[64, 128], out_channels=[64, 128], depth=1, kernel_size=9, V=25)
        self.TF4 = unit_TF_V(A,in_channels=[64,128], out_channels=[64,128,256], depth=3, kernel_size=9, V=25)
        # self.TF5 = unit_TF_V(A,in_channels=[64,128,256], out_channels=[64, 128, 256], depth=1, kernel_size=9, V=25)
        # self.TF6 = unit_TF_V(A,in_channels=[64,128,256], out_channels=[64, 128, 256], depth=1, kernel_size=9, V=25)
        # self.TF3 = unit_TF_V(in_channels=[64,128,256], out_channels=[64,128,256,512], depth=1, kernel_size=9, V=25)



        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(dim[2]),
            nn.Linear(256, num_classes)
        )

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(sum(dim)),
        #     nn.Linear(sum(dim), num_classes)
        # )

    def random_masking(self, x, mask_ratio, mmd=False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = mask_ratio*random.random() if mmd == False else mask_ratio
        len_keep = int(L * (1 - len_keep))
        # print(len_keep)

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def forward(self, x):
        x = self.cnn(x)
        x = self.TF1([x])
        # x = self.TF2(x)
        # x = self.TF3(x)
        x = self.TF4(x)
        # x = self.TF5(x)
        # x = self.TF6(x)
        x = x[-1]
        N,C,T,V = x.shape
        x = x.view(N,C,-1)
        out = x.mean(2)
        out = self.mlp_head(out)

        # x_tv_cls = x_tv[:,0,:].unsqueeze(1)
        # x_tv_mask = x_tv[:,1:,:]
        # if self.training == True:
        #     x_tv_mask = self.random_masking(x_tv_mask,mask_ratio=0.8)
        # print(x_tv.shape)
        # x_tv = torch.cat([x_tv_cls, x_tv_mask],dim = 1)

        if self.training == True:
            return out, 0.0
        else:
            return out