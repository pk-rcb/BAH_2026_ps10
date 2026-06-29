import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import TIRDataset, N_MASK_CLASSES
from models import (GlobalGenerator, SPADEGenerator,
                    MultiScaleDiscriminator, Pix2PixHDLoss)


def train_color(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {args.model.upper()}")

    use_spade = (args.model == 'spade')

    # ── Dataset ──────────────────────────────────────────────────────────────
    # SPADE task='color' returns (tir, mask_onehot, rgb)
    # Pix2Pix task='color' returns (tir, mask_onehot, rgb) — same, mask is ignored
    dataset = TIRDataset(
        patches_dir=args.patches_dir,
        task='color',
        stats_file=os.path.join(args.patches_dir, 'global_stats_color.json')
    )

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size

    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, _ = random_split(
        dataset, [train_size, val_size, test_size], generator=gen
    )
    print(f"Split → Train: {train_size} | Val: {val_size} | Test (held-out): {test_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Generator ─────────────────────────────────────────────────────────────
    if use_spade:
        generator = SPADEGenerator(
            tir_channels=1,
            label_nc=N_MASK_CLASSES,   # K=4
            out_channels=3,
            ngf=64
        ).to(device)
    else:
        generator = GlobalGenerator(
            in_channels=1, out_channels=3, ngf=64, n_blocks=9
        ).to(device)

    # Discriminator: [TIR(1ch) | RGB(3ch)] = 4ch — same for both models
    discriminator = MultiScaleDiscriminator(
        in_channels=4, ndf=64, n_layers=3, num_D=3
    ).to(device)

    g_params = sum(p.numel() for p in generator.parameters()     if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"Generator parameters:     {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")

    # ── Loss (unchanged for both models) ──────────────────────────────────────
    criterion = Pix2PixHDLoss(lambda_feat=10.0, lambda_perceptual=10.0).to(device)

    # ── Optimizers ───────────────────────────────────────────────────────────
    optimizer_G = optim.Adam(generator.parameters(),     lr=args.lr,       betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))

    # Linear LR decay starting at halfway point
    def lr_lambda(epoch):
        start_decay = args.epochs // 2
        if epoch < start_decay:
            return 1.0
        return max(0.0, 1.0 - (epoch - start_decay) / max(1, args.epochs - start_decay))

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

    os.makedirs(args.save_dir, exist_ok=True)
    best_g_val_loss = float('inf')
    save_name = f'best_{args.model}_color_model.pth'

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        total_g_loss = 0.0
        total_d_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Dataset always returns (tir, mask, rgb) for task='color'
            tir, mask, targets = batch
            tir     = tir.to(device)       # (B, 1, H, W)  TIR in [-1,1]
            mask    = mask.to(device)      # (B, K, H, W)  one-hot float
            targets = targets.to(device)   # (B, 3, H, W)  RGB in [-1,1]

            # ── Generator forward ──────────────────────────────────────────
            if use_spade:
                fake_imgs = generator(tir, mask)      # needs both TIR + mask
            else:
                fake_imgs = generator(tir)             # Pix2Pix: TIR only

            # ── Discriminator update ───────────────────────────────────────
            optimizer_D.zero_grad()
            fake_det = fake_imgs.detach()
            real_pair = torch.cat([tir, targets],  dim=1)  # (B, 4, H, W)
            fake_pair = torch.cat([tir, fake_det], dim=1)

            real_preds = discriminator(real_pair)
            fake_preds = discriminator(fake_pair)

            d_loss = criterion.discriminator_loss(real_preds, fake_preds)
            d_loss.backward()
            optimizer_D.step()

            # ── Generator update ───────────────────────────────────────────
            optimizer_G.zero_grad()
            if use_spade:
                fake_imgs = generator(tir, mask)
            else:
                fake_imgs = generator(tir)

            fake_pair  = torch.cat([tir, fake_imgs], dim=1)
            real_pair  = torch.cat([tir, targets],   dim=1)
            fake_preds = discriminator(fake_pair)
            real_preds = discriminator(real_pair)

            g_loss, g_gan, g_feat, g_perc = criterion.generator_loss(
                fake_preds, real_preds, fake_imgs, targets
            )
            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"G: {g_loss.item():.4f} (GAN:{g_gan.item():.3f} "
                      f"Feat:{g_feat.item():.3f} Perc:{g_perc.item():.3f}) "
                      f"D: {d_loss.item():.4f}")

        scheduler_G.step()
        scheduler_D.step()

        avg_g = total_g_loss / len(train_loader)
        avg_d = total_d_loss / len(train_loader)

        # ── Validation ─────────────────────────────────────────────────────
        generator.eval()
        val_g_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tir, mask, targets = batch
                tir     = tir.to(device)
                mask    = mask.to(device)
                targets = targets.to(device)

                fake_imgs  = generator(tir, mask) if use_spade else generator(tir)
                fake_pair  = torch.cat([tir, fake_imgs], dim=1)
                real_pair  = torch.cat([tir, targets],   dim=1)
                fake_preds = discriminator(fake_pair)
                real_preds = discriminator(real_pair)
                g_loss, _, _, _ = criterion.generator_loss(
                    fake_preds, real_preds, fake_imgs, targets
                )
                val_g_loss += g_loss.item()

        avg_val_g = val_g_loss / len(val_loader)
        print(f"==> Epoch [{epoch}] Train G: {avg_g:.4f} | Train D: {avg_d:.4f} | Val G: {avg_val_g:.4f}")

        if avg_val_g < best_g_val_loss:
            best_g_val_loss = avg_val_g
            save_path = os.path.join(args.save_dir, save_name)
            torch.save(generator.state_dict(), save_path)
            print(f"  Saved best {args.model} model → {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pix2PixHD or SPADE colorization model')
    parser.add_argument('--model',       type=str,   default='pix2pix',
                        choices=['pix2pix', 'spade'],
                        help='Generator architecture: pix2pix (GlobalGenerator) or spade (SPADEGenerator)')
    parser.add_argument('--patches_dir', type=str,   default='output/patches')
    parser.add_argument('--save_dir',    type=str,   default='weights')
    parser.add_argument('--batch_size',  type=int,   default=4)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int,   default=2)
    parser.add_argument('--stats_file',  type=str,   default=None)
    args = parser.parse_args()
    train_color(args)
