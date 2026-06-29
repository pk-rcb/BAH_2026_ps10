import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import TIRDataset
from models import SwinIR, SwinIRLoss


def train_sr(args):
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup Dataset
    dataset = TIRDataset(
        patches_dir=args.patches_dir,
        task='sr',
        stats_file=os.path.join(args.patches_dir, 'global_stats_sr.json')
    )

    # Train / Val / Test Split (80 / 10 / 10)
    # A fixed seed ensures the same split every run — critical for reproducible evaluation.
    # The test_dataset is held out and NEVER touched during training.
    # Report hackathon metrics against test_dataset only, after training is complete.
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size  # absorbs rounding remainders

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    print(f"Split → Train: {train_size} | Val: {val_size} | Test (held-out): {test_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # test_loader is intentionally NOT created here — evaluate after training completes.

    # Initialize SwinIR
    # input: 128x128x1 -> output: 256x256x1 (upscale=2)
    model = SwinIR(
        in_channels=1,
        out_channels=1,
        embed_dim=96,
        depths=6,
        num_heads=6,
        window_size=8,
        mlp_ratio=4.0,
        upscale=2
    ).to(device)

    print(f"SwinIR parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Charbonnier + SSIM loss (well-suited for PSNR/SSIM optimisation)
    criterion = SwinIRLoss(lambda_char=1.0, lambda_ssim=0.2).to(device)

    # AdamW with cosine LR schedule
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, l_char, l_ssim = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (Char: {l_char.item():.4f}, SSIM: {l_ssim.item():.4f})")

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss, _, _ = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"==> Epoch [{epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
              f"| LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, 'best_sr_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best SR model to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SwinIR Super-Resolution model')
    parser.add_argument('--patches_dir',  type=str,   default='output/patches', help='Path to patches directory')
    parser.add_argument('--save_dir',     type=str,   default='weights',        help='Directory to save model weights')
    parser.add_argument('--batch_size',   type=int,   default=4,                help='Batch size (reduce if OOM)')
    parser.add_argument('--epochs',       type=int,   default=100,              help='Number of epochs')
    parser.add_argument('--lr',           type=float, default=2e-4,             help='Initial learning rate')
    parser.add_argument('--num_workers',  type=int,   default=2,                help='DataLoader workers')
    # Global stats are auto-computed on first run and cached to this JSON file.
    # Delete the file to force a recompute (e.g. after adding new training data).
    parser.add_argument('--stats_file',   type=str,   default=None,
                        help='Path to cached global normalization stats JSON. '
                             'If omitted, defaults to <patches_dir>/global_stats_sr.json')

    args = parser.parse_args()
    train_sr(args)
