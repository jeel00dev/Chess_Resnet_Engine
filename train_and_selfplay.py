#!/usr/bin/env python3
"""
train_and_selfplay_fixed.py

A single-file, robust, LMDB-backed training + self-play starter script.
It fixes LMDB + DataLoader worker issues (reopening LMDB per-worker, handling
file vs directory LMDB layouts), adds a --num-workers flag, and provides safe
defaults for an RTX 4060 (8GB).

Usage examples (see bottom for more explanation):
  # supervised pretraining (recommended first)
  python train_and_selfplay_fixed.py --mode supervised --lmdb dataset.lmdb --vocab vocab.json \
      --out-dir runs/exp1 --epochs 3 --batch-size 32 --num-workers 2

  # self-play mode (seed replay from dataset and alternate self-play + training)
  python train_and_selfplay_fixed.py --mode selfplay --lmdb dataset.lmdb --vocab vocab.json \
      --out-dir runs/exp1_selfplay --mcts-sims 80 --selfplay-games 8 --num-workers 2

Dependencies:
  pip install torch torchvision python-chess lmdb numpy tqdm

Notes:
 - If you still get problems, run with --num-workers 0 to avoid multiprocessing.
 - If your LMDB is a directory (contains data.mdb, lock.mdb), it will be opened as a directory.
   If it is a single file (dataset.lmdb), it will be opened with subdir=False automatically.
"""

import os
import sys
import json
import time
import math
import random
import pickle
from argparse import ArgumentParser
from collections import deque

import lmdb
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# LMDB Dataset (robust, worker-safe)
# -----------------------------
class LMDBChessDataset(Dataset):
    """
    LMDB dataset that reopens the environment inside each worker to avoid
    segmentation faults caused by sharing LMDB env across forked workers.

    It auto-detects whether the DB path is a directory (subdir=True) or a single
    file (subdir=False) and reopens with the correct flag.
    """

    def __init__(self, lmdb_path: str):
        self.lmdb_path = os.path.abspath(lmdb_path)
        # detect 'subdir' mode: if path is a directory, use subdir=True; if it's a file, use False
        self.subdir = os.path.isdir(self.lmdb_path)
        self.env = None

        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB path not found: {self.lmdb_path}")

        # compute length safely by opening a temporary env and closing it immediately
        tmp_env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=self.subdir,
        )
        with tmp_env.begin() as txn:
            cnt = 0
            cursor = txn.cursor()
            for _ in cursor:
                cnt += 1
            self.length = cnt
        tmp_env.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        # don't pickle the env handle
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # reopen LMDB env inside worker (after fork)
        try:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=self.subdir,
            )
        except Exception as e:
            # propagate useful message
            raise RuntimeError(
                f"Failed to open LMDB in worker for path={self.lmdb_path} subdir={self.subdir}: {e}"
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # ensure env is open (in case dataset was constructed in the worker directly)
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=self.subdir,
            )

        key = f"{idx:08d}".encode("ascii")
        with self.env.begin() as txn:
            v = txn.get(key)
            if v is None:
                raise IndexError(idx)
            record = pickle.loads(v)

        board = record["board"]  # numpy (18,8,8)
        board_t = torch.from_numpy(board).to(dtype=torch.float32)

        policy = record.get("policy", None)
        if policy is not None:
            policy_t = torch.from_numpy(policy).to(dtype=torch.float32)
        else:
            idxs = record.get("policy_sparse_idx", None)
            vals = record.get("policy_sparse_val", None)
            if idxs is not None:
                policy_t = (
                    torch.from_numpy(idxs).to(torch.int64),
                    torch.from_numpy(vals).to(torch.float32),
                )
            else:
                policy_t = None

        value = float(record.get("value", 0.0))
        played_index = int(record.get("played_index", -1))
        meta = record.get("meta", {})
        return board_t, policy_t, float(value), int(played_index), meta


# -----------------------------
# Board encoder (FEN -> 18x8x8)
# -----------------------------


def encode_board(fen: str):
    b = chess.Board(fen)
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    for sq, piece in b.piece_map().items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        row = 7 - rank
        col = file
        kind = piece.piece_type
        color = piece.color
        plane_idx = (0 if color else 6) + (kind - 1)
        planes[plane_idx, row, col] = 1.0
    # castling
    planes[12, :, :] = 1.0 if b.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[13, :, :] = 1.0 if b.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if b.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[15, :, :] = 1.0 if b.has_queenside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if b.turn == chess.WHITE else 0.0
    if b.ep_square is not None:
        ep_file = chess.square_file(b.ep_square)
        planes[17, :, :] = 0.0
        planes[17, :, ep_file] = 1.0
    else:
        planes[17, :, :] = 0.0
    return planes


# -----------------------------
# Model: small ResNet trunk + policy/value heads
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu(out)
        return out


class ChessResNet(nn.Module):
    def __init__(
        self, in_channels: int, channels: int, num_blocks: int, action_size: int
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_in = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)

        # value head
        self.value_conv = nn.Conv2d(channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = self.relu(out)
        for b in self.blocks:
            out = b(out)
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = self.relu(p)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = self.relu(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.value_fc2(v)
        v = torch.tanh(v).squeeze(-1)
        return p, v


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push_many(self, examples):
        for e in examples:
            self.buffer.append(e)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        boards, policies, values = zip(*batch)
        boards = torch.tensor(np.stack(boards), dtype=torch.float32)
        policies = torch.tensor(np.stack(policies), dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        return boards, policies, values

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(list(self.buffer), fh)

    def load(self, path):
        with open(path, "rb") as fh:
            data = pickle.load(fh)
            self.buffer = deque(data, maxlen=self.capacity)


# -----------------------------
# Minimal single-process MCTS (uses network to get priors & value)
# -----------------------------
class MCTSNode:
    def __init__(self, prior=0.0):
        self.prior = float(prior)
        self.N = 0
        self.W = 0.0
        self.children = {}

    def Q(self):
        return 0.0 if self.N == 0 else (self.W / self.N)


class MCTS:
    def __init__(self, net, vocab, device, cpuct=1.0):
        self.net = net
        self.vocab = vocab
        self.idx_to_move = {v: k for k, v in vocab.items()}
        self.device = device
        self.cpuct = float(cpuct)

    def _get_priors_and_value(self, board):
        planes = encode_board(board.fen())
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
            logits = logits.squeeze(0).cpu().numpy()
            # stable softmax
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / (probs.sum() + 1e-12)
            value = float(value.item())
        legal = list(board.legal_moves)
        priors = {}
        total = 0.0
        for mv in legal:
            u = mv.uci()
            if u in self.vocab:
                p = probs[self.vocab[u]]
            else:
                p = 1e-8
            priors[u] = p
            total += p
        if total > 0:
            for k in priors:
                priors[k] /= total
        else:
            for k in priors:
                priors[k] = 1.0 / len(priors)
        return priors, value

    def run(self, root_board, num_sims=80, dirichlet_alpha=0.03, epsilon=0.25):
        root = MCTSNode()
        priors, _ = self._get_priors_and_value(root_board)
        for mv, p in priors.items():
            root.children[mv] = MCTSNode(prior=p)
        legal_moves = list(root.children.keys())
        if len(legal_moves) > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
            for i, mv in enumerate(legal_moves):
                root.children[mv].prior = (
                    root.children[mv].prior * (1 - epsilon) + noise[i] * epsilon
                )

        for _ in range(num_sims):
            node = root
            board = root_board.copy()
            path = []
            # selection
            while len(node.children) > 0:
                best_score = -float("inf")
                best_mv = None
                best_child = None
                sqrt_sum = math.sqrt(sum(c.N for c in node.children.values()) + 1e-12)
                for mv, child in node.children.items():
                    U = self.cpuct * child.prior * sqrt_sum / (1 + child.N)
                    score = child.Q() + U
                    if score > best_score:
                        best_score = score
                        best_mv = mv
                        best_child = child
                if best_mv is None:
                    break
                board.push(chess.Move.from_uci(best_mv))
                path.append((node, best_mv))
                node = best_child
                if node.N == 0:
                    break
            # expansion/eval
            if board.is_game_over():
                res = board.result()
                if res == "1-0":
                    value = 1.0
                elif res == "0-1":
                    value = -1.0
                else:
                    value = 0.0
            else:
                priors, value = self._get_priors_and_value(board)
                for mv, p in priors.items():
                    if mv not in node.children:
                        node.children[mv] = MCTSNode(prior=p)
            # backprop
            for parent, mv in reversed(path):
                child = parent.children[mv]
                child.N += 1
                child.W += value
                value = -value
        visits = {mv: child.N for mv, child in root.children.items()}
        total_visits = sum(visits.values())
        if total_visits > 0:
            policy = {mv: n / total_visits for mv, n in visits.items()}
        else:
            policy = {mv: 1.0 / len(visits) for mv in visits}
        return policy


# -----------------------------
# Training helpers
# -----------------------------


def supervised_train(
    model,
    optimizer,
    scaler,
    dataloader,
    epoch,
    device,
    value_coeff=1.0,
    log_interval=200,
):
    model.train()
    total_loss = 0.0
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    mse_loss = nn.MSELoss()
    pbar = tqdm(
        enumerate(dataloader), total=len(dataloader), desc=f"Train epoch {epoch}"
    )
    for i, batch in pbar:
        try:
            boards, policy, values, played_idx, meta = batch
        except Exception:
            # compatibility: some DataLoader collate behaviors
            boards, policy, values, played_idx, meta = batch
        if boards.dim() == 3:
            boards = boards.unsqueeze(0)
        boards = boards.to(device)
        bs = boards.size(0)
        # Prepare target policy (dense)
        action_dim = model.policy_fc.out_features
        if isinstance(policy, torch.Tensor):
            target_policy = policy.to(device)
            if target_policy.dim() == 1:
                target_policy = target_policy.unsqueeze(0)
        else:
            # fallback: build one-hot from played_idx
            target_policy = torch.zeros(
                (bs, action_dim), dtype=torch.float32, device=device
            )
            for bi in range(bs):
                pi = (
                    int(played_idx[bi])
                    if isinstance(played_idx[bi], (int, np.integer))
                    else int(played_idx[bi].item())
                )
                if pi >= 0 and pi < action_dim:
                    target_policy[bi, pi] = 1.0

        target_value = torch.tensor(values, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits, preds = model(boards)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = kl_loss(log_probs, target_policy)
            value_loss = mse_loss(preds, target_value)
            loss = policy_loss + value_coeff * value_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
        if i % log_interval == 0:
            pbar.set_postfix({"loss": total_loss / (i + 1)})
    return total_loss / max(1, len(dataloader))


# -----------------------------
# Self-play generation (simple)
# -----------------------------


def run_selfplay_games(net, vocab, device, num_games=4, sims=80, cpuct=1.0):
    mcts = MCTS(net, vocab, device, cpuct=cpuct)
    examples = []
    for g in range(num_games):
        board = chess.Board()
        game_examples = []
        while not board.is_game_over():
            policy_probs = mcts.run(board, num_sims=sims)
            action_dim = len(vocab)
            pol_vec = np.zeros((action_dim,), dtype=np.float32)
            for mv, p in policy_probs.items():
                if mv in vocab:
                    pol_vec[vocab[mv]] = p
            game_examples.append((encode_board(board.fen()), pol_vec, None))
            moves = list(policy_probs.keys())
            probs = np.array([policy_probs[m] for m in moves], dtype=np.float64)
            probs = probs / (probs.sum() + 1e-12)
            idx = np.random.choice(len(moves), p=probs)
            mv = moves[idx]
            board.push(chess.Move.from_uci(mv))
        res = board.result()
        if res == "1-0":
            z = 1.0
        elif res == "0-1":
            z = -1.0
        else:
            z = 0.0
        for bplanes, pol, _ in game_examples:
            examples.append((bplanes, pol, z))
    return examples


# -----------------------------
# Checkpoint utils
# -----------------------------


def save_checkpoint(path, model, optimizer, scaler, epoch, extra=None):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    try:
        state["scaler_state_dict"] = scaler.state_dict()
    except Exception:
        pass
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception:
            pass
    return ckpt.get("epoch", 0)


# -----------------------------
# Main driver
# -----------------------------


def main():
    ap = ArgumentParser()
    ap.add_argument("--mode", choices=["supervised", "selfplay"], default="supervised")
    ap.add_argument(
        "--lmdb", required=True, help="Path to dataset.lmdb (file) or LMDB dir"
    )
    ap.add_argument("--vocab", required=True, help="vocab.json mapping move->index")
    ap.add_argument("--out-dir", default="runs/exp")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mcts-sims", type=int, default=80)
    ap.add_argument("--selfplay-games", type=int, default=8)
    ap.add_argument("--replay-size", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # load vocab
    with open(args.vocab, "r", encoding="utf-8") as fh:
        vocab = json.load(fh)
    action_size = len(vocab)
    print(f"Loaded vocab size {action_size}")

    # model
    model = ChessResNet(
        in_channels=18,
        channels=args.channels,
        num_blocks=args.blocks,
        action_size=action_size,
    )
    model.to(DEVICE)
    print(
        "Model params:", sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # use amp scaler
    try:
        scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
    except Exception:
        # fallback
        scaler = torch.cuda.amp.GradScaler()

    if args.mode == "supervised":
        dataset = LMDBChessDataset(args.lmdb)

        def worker_init_fn(worker_id):
            # re-seed worker RNGs and ensure LMDB env open
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                return
            ds = worker_info.dataset
            # reopen env if needed
            if getattr(ds, "env", None) is None:
                ds.env = lmdb.open(
                    ds.lmdb_path,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    subdir=ds.subdir,
                )
            seed = args.seed + worker_id
            np.random.seed(seed)
            random.seed(seed)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=(args.num_workers > 0),
        )

        for epoch in range(1, args.epochs + 1):
            loss = supervised_train(model, optimizer, scaler, loader, epoch, DEVICE)
            print(f"Epoch {epoch} average loss: {loss:.4f}")
            ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, epoch)
            print("Saved checkpoint to", ckpt_path)

    elif args.mode == "selfplay":
        # seed replay from dataset
        replay = ReplayBuffer(capacity=args.replay_size)
        dataset = LMDBChessDataset(args.lmdb)
        seed_n = min(20000, len(dataset)) if len(dataset) > 0 else 0
        if seed_n > 0:
            print(f"Seeding replay with {seed_n} supervised positions...")
            idxs = np.random.choice(len(dataset), seed_n, replace=False)
            for i in tqdm(idxs):
                b, pol, v, played, meta = dataset[i]
                if isinstance(pol, torch.Tensor):
                    pol_np = pol.numpy()
                else:
                    pol_np = np.zeros((action_size,), dtype=np.float32)
                    if played >= 0:
                        pol_np[played] = 1.0
                replay.push_many([(b.numpy(), pol_np, v)])
            print("Replay seeded. size=", len(replay))

        batch_size = args.batch_size
        iteration = 0
        while True:
            iteration += 1
            print(
                f"Self-play iter {iteration}: generate {args.selfplay_games} games (sims={args.mcts_sims})"
            )
            new_examples = run_selfplay_games(
                model, vocab, DEVICE, num_games=args.selfplay_games, sims=args.mcts_sims
            )
            print("Generated", len(new_examples), "positions")
            replay.push_many(new_examples)
            print("Replay size", len(replay))

            # train small amount
            train_steps = 200
            for step in range(train_steps):
                if len(replay) < batch_size:
                    break
                boards, policies, values = replay.sample(batch_size)
                boards = boards.to(DEVICE)
                policies = policies.to(DEVICE)
                values = values.to(DEVICE)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                    logits, preds = model(boards)
                    log_probs = F.log_softmax(logits, dim=1)
                    policy_loss = nn.KLDivLoss(reduction="batchmean")(
                        log_probs, policies
                    )
                    value_loss = nn.MSELoss()(preds, values)
                    loss = policy_loss + value_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if iteration % 5 == 0:
                ckpt_path = os.path.join(args.out_dir, f"selfplay_iter{iteration}.pt")
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    scaler,
                    iteration,
                    extra={"replay_size": len(replay)},
                )
                print("Saved checkpoint to", ckpt_path)

            # stop condition for demo
            if iteration >= 1000:
                break


if __name__ == "__main__":
    main()
