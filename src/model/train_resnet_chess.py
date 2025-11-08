#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================
# Dataset
# ============================================


class ChessTensorDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)

        # Force correct dtypes (this fixes the dtype error you had)
        self.boards = data["boards"].float()  # (N,12,8,8) float32
        self.move_index = data["move_index"].long()  # (N,) int64 for CE
        self.eval = data["eval"].float().squeeze(-1)  # (N,) float32

        assert self.boards.shape[0] == self.move_index.shape[0] == self.eval.shape[0]

    def __len__(self):
        return self.boards.shape[0]

    def __getitem__(self, idx):
        return self.boards[idx], self.move_index[idx], self.eval[idx]


# ============================================
# Model: ResNet Trunk + Policy Head + Value Head
# ============================================


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)  # residual connection
        return out


class ResNetChess(nn.Module):
    def __init__(self, channels=128, blocks=6):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # Policy head → 4096 classes (from*64 + to)
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)

        # Value head → scalar evaluation in [0,1]
        self.value = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(16 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.layers(x)

        # Policy path
        p = self.policy(x)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)  # raw logits → softmax in loss

        # Value path
        v = self.value(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.sigmoid(self.value_fc2(v)).squeeze(-1)  # bound to [0,1]

        return policy_logits, v


# ============================================
# Training
# ============================================


def train_one_epoch(model, loader, optimizer, device, value_coef):
    model.train()
    total, correct = 0, 0
    mse_sum = 0.0
    total_loss_sum = 0.0

    for boards, move_idx, eval_target in loader:
        boards = boards.to(device)
        move_idx = move_idx.to(device)
        eval_target = eval_target.to(device)

        optimizer.zero_grad()
        logits, value_pred = model(boards)

        # LOSS:
        policy_loss = F.cross_entropy(logits, move_idx)
        value_loss = F.mse_loss(value_pred, eval_target)

        loss = policy_loss + value_coef * value_loss
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item() * boards.size(0)

        # Policy accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == move_idx).sum().item()
        total += boards.size(0)

        # Value loss metric
        mse_sum += ((value_pred - eval_target) ** 2).sum().item()

    return {
        "loss": total_loss_sum / total,
        "policy_acc": correct / total,
        "value_mse": mse_sum / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ds = ChessTensorDataset(args.data)
    N = len(ds)
    train_size = int(0.95 * N)
    val_size = N - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = ResNetChess().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        stats = train_one_epoch(model, train_loader, optimizer, device, args.value_coef)
        print(
            f"Epoch {epoch:03d} | loss={stats['loss']:.4f} | policy_acc={stats['policy_acc']:.4f} | value_mse={stats['value_mse']:.6f}"
        )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            f"{args.out_dir}/last.pt",
        )


if __name__ == "__main__":
    main()
