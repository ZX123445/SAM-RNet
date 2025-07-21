import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def build_grid(resolution):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranges = [np.linspace(0., 1., num=res) for res in resolution]#为给定的分辨率（resolution）生成从0到1的等间距值列表。
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")#使用np.meshgrid生成一个多维网格
    grid = np.stack(grid, axis=-1) #将网格的各个维度堆叠到最后一个维度上。
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])#重新调整网格形状，使其具有指定的分辨率和堆叠的维度。
    grid = np.expand_dims(grid, axis=0) #在第一个维度上增加一个维度
    grid = grid.astype(np.float32) #将网格类型转化为float32
    return torch.tensor(np.concatenate([grid, 1.0 - grid], axis=-1)) #在最后一个维度上拼接网格和其补集， 然后转化为pythorch张量


class SoftPositionEmbed(nn.Module):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, hidden_size, resolution):
    """Builds the soft position embedding layer.

    Args:
      hidden_size: Size of input feature dimension.
      resolution: Tuple of integers specifying width and height of grid.
    """
    super(SoftPositionEmbed, self).__init__()
    self.proj = nn.Linear(4, hidden_size) #线性层， 将4维位置嵌入投影到隐藏层大小
    self.grid = build_grid(resolution)#生成位置网格

  def forward(self, inputs):
    device = inputs.device
    self.grid = self.grid.to(device)
    return inputs + self.proj(self.grid)  #将位置嵌入投影到隐藏层大小，并与输入特征相加。


def spatial_broadcast(slots, resolution):#将槽特征广播到2D网格上
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    # slots = torch.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]

    slots = torch.reshape(slots, [-1, slots.shape[-2],slots.shape[-1]])[:, :,None, None, :]
    grid = einops.repeat(slots, 'b n i j d -> b n (tilei i) (tilej j) d', tilei=resolution[0], tilej=resolution[1])
    # `grid` has shape: [batch_size*num_slots, height, width, slot_size].
    return grid

def spatial_broadcast2(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    slots = torch.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
    grid = einops.repeat(slots, 'b_n i j d -> b_n (tilei i) (tilej j) d', tilei=resolution[0], tilej=resolution[1])
    # `grid` has shape: [batch_size*num_slots, height, width, slot_size].
    return grid

def spatial_flatten(x): #将输入张量x的空间维度展平
  return torch.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])


def unstack_and_split(x, batch_size, num_channels=3): #将输入张量x的批处理维度展开，并将其分割为通道和alpha掩码。
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = einops.rearrange(x, '(b s) c h w -> b s c h w', b=batch_size)
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=2)
    return channels, masks


class SlotAttention(nn.Module):#槽注意   MFG模块
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.槽特征向量的维度
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.归一化前注意系数的偏移量
        """
        super(SlotAttention, self).__init__()

        self.eps = eps
        self.iters = iters #迭代次数
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        # Parameters for Gaussian init (shared by all slots).
        # self.slots_mu = nn.Parameter(torch.randn(1, 1, encoder_dims))
        # self.slots_sigma = nn.Parameter(torch.randn(1, 1, encoder_dims))

        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)#槽嵌入层（槽数目， 编码维度）

        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        # Slot update functions.
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)#定义一个GRU单元用于槽更新

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

    def forward(self, inputs, num_slots=None):  #输入特征和可选的槽数量
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # random slots initialization,
        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_sigma.expand(b, n_s, -1)
        # slots = torch.normal(mu, sigma)

        # learnable slots initialization
        slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))

        # Multiple rounds of attention.
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

            updates = torch.einsum('bjd,bij->bid', v, attn)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots


class SlotAttentionAutoEncoder(nn.Module):
    """Slot Attention-based auto-encoder for object discovery."""
    def __init__(self, resolution, num_slots, in_out_channels=3, iters=5):
        """Builds the Slot Attention-based Auto-encoder.

        Args:
            resolution: Tuple of integers specifying width and height of input image
            num_slots: Number of slots in Slot Attention.
            iters: Number of iterations in Slot Attention.
        """
        super(SlotAttentionAutoEncoder, self).__init__()

        self.iters = iters
        self.num_slots = num_slots
        self.resolution = resolution
        self.in_out_channels = in_out_channels

        self.encoder_arch = [64, 'MP', 128, 'MP', 256]#编码架构， 定义了卷积层的数量和类型（MP；最大池化）
        self.encoder_dims = self.encoder_arch[-1]

        #make_encoder函数根据encoder_arch构建卷积神经网络（CNN）编码器，并返回编码器和下采样因子（ratio）
        self.encoder_cnn, ratio = self.make_encoder(self.in_out_channels, self.encoder_arch)

        self.encoder_end_size = (int(resolution[0] / ratio), int(resolution[1] / ratio))

        #位置嵌入（encoder_pos和decoder_pos）用于向编码器和解码器的输入添加位置信息
        self.encoder_pos = SoftPositionEmbed(self.encoder_dims, self.encoder_end_size)
        self.decoder_initial_size = (int(resolution[0] / 8), int(resolution[1] / 8))
        self.decoder_pos = SoftPositionEmbed(self.encoder_dims, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(self.encoder_dims)

        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_dims, self.encoder_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dims, self.encoder_dims)
        )

        #SlotAttention将输入图像分解为多个槽， 每个槽代表图像中的一个潜在对象或组件
        self.slot_attention = SlotAttention(
            iters=self.iters,
            num_slots=self.num_slots,
            encoder_dims=self.encoder_dims,
            hidden_dim=self.encoder_dims)

        #转置卷积，用于将槽的表示转换回图像空间
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_dims, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2, output_padding=1, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_out_channels + 1, kernel_size=5, padding=2, stride=1)
        )

    def make_encoder(self, in_channels, encoder_arch):
        #encoder_arch:是一个列表， 定义了编码器的结构，包括要使用的卷积层的输出通道数和是否应用最大池化层
        layers = [] #用于存储编码器的所有层
        down_factor = 0 #用于记录由于最大池化操作导致的空间分辨率下降的因子
        for v in encoder_arch:
            if v == 'MP':
                layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                down_factor += 1
            else:
                conv1 = nn.Conv2d(in_channels, v, kernel_size=5, padding=2)
                conv2 = nn.Conv2d(v, v, kernel_size=5, padding=2)

                layers += [conv1, nn.InstanceNorm2d(v, affine=True), nn.ReLU(inplace=True),
                           conv2, nn.InstanceNorm2d(v, affine=True), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers), 2 ** down_factor


    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, height, width].
        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.encoder_pos(x)  # Position embedding.
        x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
        x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # Spatial broadcast decoder.
        x = spatial_broadcast(slots, self.decoder_initial_size)
        # `x` has shape: [batch_size*num_slots, height_init, width_init, slot_size].
        x = self.decoder_pos(x)
        x = einops.rearrange(x, 'b_n h w c -> b_n c h w')
        x = self.decoder_cnn(x)
        # `x` has shape: [batch_size*num_slots, num_channels+1, height, width].
        
        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = unstack_and_split(x, batch_size=image.shape[0], num_channels=self.in_out_channels)
        # `recons` has shape: [batch_size, num_slots, num_channels, height, width].
        # `masks` has shape: [batch_size, num_slots, 1, height, width].
        
        # Normalize alpha masks over slots.
        masks = torch.softmax(masks, axis=1)

        recon_combined = torch.sum(recons * masks, axis=1)  # Recombine image.
        # `recon_combined` has shape: [batch_size, num_channels, height, width].
        return recon_combined, recons, masks, slots


if __name__ =='__main__':
    x = torch.rand(4, 128, 12, 12).cuda()
    print(x.shape)

    x = einops.rearrange(x, 'b c h w -> b h w c')
    print(x.shape)

    encoder_pos = SoftPositionEmbed(128, (int(12), int(12))).cuda()
    x = encoder_pos(x)  # Position embedding.
    print(x.shape)

    x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
    print(x.shape)

    layer_norm = nn.LayerNorm(128).cuda()
    mlp = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128)
    ).cuda()
    x = mlp(layer_norm(x))
    print(x.shape)
    print('*'*10)

    net = SlotAttention(num_slots=4, encoder_dims=128).cuda()
    y = net(x)
    print(y.shape)
    
    x = spatial_broadcast(y, (int(12), int(12)))
    print(x.shape)
    print('*'*10)
    decoder_pos = SoftPositionEmbed(128, (int(12), int(12))).cuda()
    x = decoder_pos(x)
    print(x.shape)
    x = einops.rearrange(x, 'b n h w c -> b (n c) h w')
    print(x.shape)


