import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==========================================
# 1. SWINIR - Vision Transformer for Super-Resolution
# ==========================================

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """Partition into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad to be divisible by window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._compute_attn_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Crop padding
        x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def _compute_attn_mask(self, H, W, device):
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class RSTB(nn.Module):
    """Residual Swin Transformer Block (a group of SwinTransformerBlocks + a conv)."""
    def __init__(self, dim, num_heads, depth=6, window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, H, W):
        shortcut = x
        for blk in self.blocks:
            x = blk(x, H, W)
        x = x.transpose(1, 2).view(-1, x.shape[-1], H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x + shortcut


class SwinIR(nn.Module):
    """
    SwinIR for Super-Resolution.
    in_channels: 1 (single TIR band)
    upscale: 2 (100m -> 50m equivalent or 200m -> 100m)
    """
    def __init__(self, in_channels=1, out_channels=1, embed_dim=96, depths=6,
                 num_heads=6, window_size=8, mlp_ratio=4.0, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.window_size = window_size

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Deep feature extraction
        self.rstb = RSTB(dim=embed_dim, num_heads=num_heads, depth=depths, window_size=window_size, mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)

        # Reconstruction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(embed_dim, out_channels, 3, 1, 1)
        )

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.check_image_size(x)
        _, _, Hp, Wp = x.shape

        # Shallow features
        feat = self.conv_first(x)

        # Deep features (RSTB operates on flattened tokens)
        B, C, H2, W2 = feat.shape
        feat_seq = feat.flatten(2).transpose(1, 2)  # B, H*W, C
        feat_seq = self.rstb(feat_seq, H2, W2)
        feat_seq = self.norm(feat_seq)
        deep_feat = feat_seq.transpose(1, 2).view(B, C, H2, W2)

        # Residual connection
        deep_feat = self.conv_after_body(deep_feat) + feat

        # Upsampling
        out = self.upsample(deep_feat)

        # Crop to expected output size
        out = out[:, :, :H * self.upscale, :W * self.upscale]
        return torch.tanh(out)


# ==========================================
# 2. PIX2PIXHD - Advanced GAN for Colorization
# ==========================================

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class GlobalGenerator(nn.Module):
    """
    Pix2PixHD Global Generator.
    in_channels: 1 (TIR band)
    out_channels: 3 (RGB)
    ngf: base number of filters
    n_blocks: number of residual blocks
    """
    def __init__(self, in_channels=1, out_channels=3, ngf=64, n_blocks=9):
        super().__init__()

        # Encoder
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        # Downsampling
        n_downsampling = 4
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]

        # Residual Blocks
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        # Decoder (Upsampling)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator for a single scale."""
    def __init__(self, in_channels, ndf=64, n_layers=3, get_intermediate=True):
        super().__init__()
        self.get_intermediate = get_intermediate

        sequence = [
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=2),
            nn.LeakyReLU(0.2, True)
        ]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                nn.Conv2d(nf_prev, nf, 4, stride=2, padding=2),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, True)
            ]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            nn.Conv2d(nf_prev, nf, 4, stride=1, padding=2),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, 1, 4, stride=1, padding=2)
        ]
        # Build as ModuleList for intermediate feature extraction
        self.model = nn.ModuleList()
        for layer in sequence:
            self.model.append(layer)

    def forward(self, x):
        feats = []
        for layer in self.model:
            x = layer(x)
            feats.append(x)
        if self.get_intermediate:
            return feats  # list of feature maps
        else:
            return [x]


class MultiScaleDiscriminator(nn.Module):
    """
    Pix2PixHD Multi-Scale Discriminator.
    Uses 3 discriminators at different scales.
    in_channels: condition + output channels (e.g. 1 + 3 = 4)
    """
    def __init__(self, in_channels=4, ndf=64, n_layers=3, num_D=3):
        super().__init__()
        self.num_D = num_D
        self.discriminators = nn.ModuleList()
        for _ in range(num_D):
            self.discriminators.append(NLayerDiscriminator(in_channels, ndf, n_layers))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        results = []
        for i, disc in enumerate(self.discriminators):
            results.append(disc(x))
            if i != self.num_D - 1:
                x = self.downsample(x)
        return results  # list of lists of feature maps


# ==========================================
# 3. LOSS FUNCTIONS
# ==========================================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean() if self.size_average else 1 - ssim_map.mean(1).mean(1).mean(1)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(16)])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        x = (x + 1) / 2
        y = (y + 1) / 2
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std
        return F.l1_loss(self.slice1(x), self.slice1(y))


class SwinIRLoss(nn.Module):
    """Combined Charbonnier + SSIM loss for SwinIR super-resolution."""
    def __init__(self, lambda_char=1.0, lambda_ssim=0.2):
        super().__init__()
        self.char_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim

    def forward(self, output, target):
        loss_char = self.char_loss(output, target)
        loss_ssim = self.ssim_loss(output, target)
        total = self.lambda_char * loss_char + self.lambda_ssim * loss_ssim
        return total, loss_char, loss_ssim


class Pix2PixHDLoss(nn.Module):
    """
    Pix2PixHD adversarial + feature-matching + perceptual loss for the Generator.
    """
    def __init__(self, lambda_feat=10.0, lambda_perceptual=10.0):
        super().__init__()
        self.lambda_feat = lambda_feat
        self.lambda_perceptual = lambda_perceptual
        self.perc_loss = PerceptualLoss()
        self.feat_loss = nn.L1Loss()
        self.gan_loss = nn.MSELoss()  # LSGAN

    def discriminator_loss(self, real_preds, fake_preds):
        """Hinge-style LSGAN discriminator loss over all scales."""
        d_loss = 0.0
        for real_scale, fake_scale in zip(real_preds, fake_preds):
            real_pred = real_scale[-1]
            fake_pred = fake_scale[-1]
            d_loss += self.gan_loss(real_pred, torch.ones_like(real_pred))
            d_loss += self.gan_loss(fake_pred, torch.zeros_like(fake_pred))
        return d_loss

    def generator_loss(self, fake_preds, real_preds, fake_img, real_img):
        """Generator loss: GAN + feature matching + perceptual."""
        # GAN loss (fool the discriminator)
        g_gan = 0.0
        for fake_scale in fake_preds:
            pred = fake_scale[-1]
            g_gan += self.gan_loss(pred, torch.ones_like(pred))

        # Feature matching loss
        g_feat = 0.0
        for real_scale, fake_scale in zip(real_preds, fake_preds):
            for real_feat, fake_feat in zip(real_scale[:-1], fake_scale[:-1]):
                g_feat += self.feat_loss(fake_feat, real_feat.detach())

        # Perceptual loss
        g_perc = self.perc_loss(fake_img, real_img)

        total = g_gan + self.lambda_feat * g_feat + self.lambda_perceptual * g_perc
        return total, g_gan, g_feat, g_perc


# ==========================================
# 4. LEGACY: Multi-Task Loss (kept for compatibility)
# ==========================================

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_char=1.0, lambda_ssim=0.1, lambda_perceptual=0.1):
        super().__init__()
        self.char_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.perc_loss = PerceptualLoss()
        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual

    def forward(self, output, target):
        loss_char = self.char_loss(output, target)
        loss_ssim = self.ssim_loss(output, target)
        loss_perc = self.perc_loss(output, target)
        total_loss = (self.lambda_char * loss_char) + \
                     (self.lambda_ssim * loss_ssim) + \
                     (self.lambda_perceptual * loss_perc)
        return total_loss, loss_char, loss_ssim, loss_perc


# ==========================================
# 5. SPADE — Spatially-Adaptive Normalization (GauGAN)
# ==========================================

class SPADE(nn.Module):
    """
    Spatially-Adaptive Denormalization layer.

    Replaces InstanceNorm's fixed affine params with spatial params learned
    from the semantic mask:  out = gamma(seg) * InstanceNorm(x) + beta(seg)

    Args:
        norm_nc  : channels of the feature map to normalise
        label_nc : channels of the one-hot segmentation mask (= K classes)
        nhidden  : hidden channels in the mask-to-params convnet (default 128)
    """
    def __init__(self, norm_nc: int, label_nc: int, nhidden: int = 128):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, 3, padding=1),
            nn.ReLU(True),
        )
        self.gamma = nn.Conv2d(nhidden, norm_nc, 3, padding=1)
        self.beta  = nn.Conv2d(nhidden, norm_nc, 3, padding=1)

    def forward(self, x: 'torch.Tensor', seg: 'torch.Tensor') -> 'torch.Tensor':
        seg_r  = F.interpolate(seg, size=x.shape[2:], mode='nearest')
        normed = self.param_free_norm(x)
        h      = self.shared(seg_r)
        return normed * (1.0 + self.gamma(h)) + self.beta(h)


class SPADEResBlock(nn.Module):
    """Residual block using SPADE normalisation + spectral norm convolutions."""
    def __init__(self, fin: int, fout: int, label_nc: int):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmid = min(fin, fout)
        sn = nn.utils.spectral_norm

        self.conv_0 = sn(nn.Conv2d(fin,  fmid, 3, padding=1))
        self.conv_1 = sn(nn.Conv2d(fmid, fout, 3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sn(nn.Conv2d(fin, fout, 1, bias=False))

        self.norm_0 = SPADE(fin,  label_nc)
        self.norm_1 = SPADE(fmid, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def _shortcut(self, x, seg):
        return self.conv_s(self.norm_s(x, seg)) if self.learned_shortcut else x

    def forward(self, x: 'torch.Tensor', seg: 'torch.Tensor') -> 'torch.Tensor':
        x_s = self._shortcut(x, seg)
        dx  = self.conv_0(F.leaky_relu(self.norm_0(x,  seg), 0.2, inplace=True))
        dx  = self.conv_1(F.leaky_relu(self.norm_1(dx, seg), 0.2, inplace=True))
        return x_s + dx


class SPADEGenerator(nn.Module):
    """
    SPADE / GauGAN Generator — drop-in upgrade for GlobalGenerator.

    Inputs:
        tir : (B, 1, H, W)  thermal image in [-1, 1]  — provides texture/style
        seg : (B, K, H, W)  one-hot semantic mask float — provides spatial layout

    Output:
        rgb : (B, 3, H, W)  synthesised RGB in [-1, 1]
        NOTE: training uses RGB order. inference.py swaps to BGR before saving .tif.

    Architecture  (for 256x256 output):
        TIR style encoder  ->  (B, 512, 4, 4)
        6 x (upsample x2 + SPADEResBlock) -> (B, 64, 256, 256)
        3x3 conv + Tanh -> (B, 3, 256, 256)

    Channel schedule: 512->512->512->256->128->64->64

    Discriminator (MultiScaleDiscriminator) and losses (Pix2PixHDLoss)
    are UNCHANGED — they still receive [TIR | RGB] = 4ch concatenated input.
    """
    def __init__(self, tir_channels: int = 1, label_nc: int = 4,
                 out_channels: int = 3, ngf: int = 64):
        super().__init__()

        # TIR Style Encoder: (B,1,256,256) -> (B,512,4,4)
        self.style_enc = nn.Sequential(
            nn.Conv2d(tir_channels,  64, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d( 64, 128, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(4),
        )

        # Progressive SPADE upsampling: 4->8->16->32->64->128->256
        ch = [512, 512, 512, 256, 128, 64, 64]
        self.up_blocks = nn.ModuleList([
            SPADEResBlock(ch[i], ch[i + 1], label_nc)
            for i in range(len(ch) - 1)
        ])

        self.conv_out = nn.Conv2d(ch[-1], out_channels, 3, padding=1)

    def forward(self, tir: 'torch.Tensor', seg: 'torch.Tensor') -> 'torch.Tensor':
        x = self.style_enc(tir)                          # (B, 512, 4, 4)
        for block in self.up_blocks:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = block(x, seg)
        return torch.tanh(self.conv_out(x))
