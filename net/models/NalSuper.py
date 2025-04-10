import torch.nn as nn
import torch
# import net.networks as networks
import clip
from torchvision.transforms import Resize
from torch import einsum
from einops import rearrange, repeat
import torch.nn.functional as F
from inspect import isfunction

'''CLIP code'''
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP, preprocess = clip.load("RN50", device=device)
text = clip.tokenize(["high light image"]).to(device)




def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)
    
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y



# TODO
#### Cross-layer Attention Fusion Block
class LAM_Module_v2(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim,bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize,N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out



 # TODO
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=8, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class DSC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DSC, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out







class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y






# TODO

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()    # dim =64
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
        # TODO
        self.text_feature = text
        self.layer_fussion = LAM_Module_v2(in_dim=int(dim * 3))
        self.conv_fuss = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=False)
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = zero_module(nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=0., glu=True)
        self.cross_attention = CrossAttention(dim, context_dim=dim * 8)

# TODO IFA forward

#     def forward(self, x):
#         x_in = x
#         res=self.act1(self.conv1(x))
#         fussion_1 = res
#         res=self.conv2(res)
#         res=self.calayer(res)
#         res=self.palayer(res)
#         # TODO
#         res += x
#         fussion_2 = res
#         inp_fusion_123 = torch.cat(
#              [x.unsqueeze(1), fussion_1.unsqueeze(1), fussion_2.unsqueeze(1)], dim=1)
#         out_fusion_123 = self.layer_fussion(inp_fusion_123)
#         out_fusion_123 = self.conv_fuss(out_fusion_123)
#         return out_fusion_123 +x_in

# TODO TCM forward
#
#     def forward(self, x):
#         x_in = x
#         context = CLIP(self.text_feature)
#         context = context.float()
#         b, c, h, w = x.shape
#         res = rearrange(x, 'b c h w -> b (h w) c').contiguous()
#         res = self.proj_in(res)
#         res = self.cross_attention(self.norm(res), context) + res
#         res = self.ff(self.norm(res)) + res
#         res = self.proj_out(res)
#         res = rearrange(res, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
#         return res + x_in
#
    #
    def forward(self, x):
        # TODO  context_process
        context = CLIP(self.text_feature)
        context = context.float()
        b, c, h, w = x.shape
        x_in = x
        res=self.act1(self.conv1(x))
        # TODO
        res=res+x
        # TODO
        fussion_1 = res
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        # TODO
        res += x
        fussion_2 = res
        inp_fusion_123 = torch.cat(
             [x.unsqueeze(1), fussion_1.unsqueeze(1), fussion_2.unsqueeze(1)], dim=1)
        out_fusion_123 = self.layer_fussion(inp_fusion_123)
        out_fusion_123 = self.conv_fuss(out_fusion_123)

        res = rearrange(out_fusion_123, 'b c h w -> b (h w) c').contiguous()
        res = self.proj_in(res)


        res = self.cross_attention(self.norm(res), context) + res
        res = self.ff(self.norm(res)) + res
        res = self.proj_out(res)
        res = rearrange(res, 'b (h w) c -> b c h w', h=h, w=w).contiguous()

        # return out_fusion_123
        return res + x_in


# TODO
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [ Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res= self.gp(x)
        res += x
        return res

class NalSuper(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(NalSuper, self).__init__()
        self.gps=gps
        self.dim=64
        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g2= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g3= Group(conv, self.dim, kernel_size,blocks=blocks)
        # TODO
        # self.layer_fussion = LAM_Module_v2(in_dim=int(self.dim * 3))
        # self.conv_fuss = nn.Conv2d(int(self.dim * 3), int(self.dim * 3), kernel_size=1, bias=False)
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer=PALayer(self.dim)
        # TODO 后处理维度映射
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]
        # TODO 注释
        self.backbone = nn.Sequential(*[
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim, self.dim, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim, self.dim, 1, padding=0),
            nn.Sigmoid()
        ])
        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

        # TODO
        self.layer_fussion = LAM_Module_v2(in_dim=int(self.dim * 3))
        self.conv_fuss = nn.Conv2d(int(self.dim * 3), int(self.dim * 3), kernel_size=1, bias=False)

    def forward(self, x1):
        x = self.pre(x1)
        # a = x.clone()
        # b = x.clone()
        # c = x.clone()
        res1=self.g1(x)
        res2=self.g2(res1)
        res3=self.g3(res2)

        # inp_fusion_123 = torch.cat(
        #     [res1.unsqueeze(1), res2.unsqueeze(1), res3.unsqueeze(1)], dim=1)
        # out_fusion_123 = self.layer_fussion(inp_fusion_123)
        # out_fusion_123 = self.conv_fuss(out_fusion_123)
        # w = self.ca(out_fusion_123)
        w=self.ca(torch.cat([res1, res2, res3],dim=1))
        # TODO ablation
        # w = self.ca(torch.cat([a, b, c], dim=1))

        w=w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out=w[:, 0, ::]*res1+w[:, 1, ::]*res2+w[:, 2, ::]*res3

        # out = w[:, 0, ::] * a + w[:, 1, ::] * b + w[:, 2, ::] * c
        out=self.palayer(out)
        x=self.post(out)

        return x
if __name__ == "__main__":
    net=NalSuper(gps=3,blocks=19)
    print(net)