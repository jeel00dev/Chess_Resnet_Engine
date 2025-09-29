#!/usr/bin/env python3
"""
train_and_selfplay.py

python train_and_selfplay.py --mode supervised --lmdb dataset.lmdb --vocab vocab.json --out-dir runs/exp1 --epochs 5 --batch-size 32
python train_and_selfplay.py --mode selfplay --lmdb dataset.lmdb --vocab vocab.json --out-dir runs/exp1_selfplay --mcts-sims 80 --selfplay-games 8 --batch-size 32

Large single-file script that provides:
 - LMDB dataset reader (expects dataset created by jsonl_to_lmdb.py)
 - ResNet model (trunk + policy & value heads)
 - Supervised pretraining loop (policy + value)
 - Simple AlphaZero-style MCTS self-play (single-process)
 - Replay buffer for storing self-play examples
 - Checkpoint saving & loading

Designed for a single GPU (RTX 4060, 8GB). Sensible defaults chosen for that hardware.

Dependencies:
    pip install torch torchvision python-chess lmdb numpy tqdm

Usage examples:
  # supervised pretraining only (assumes dataset.lmdb + vocab.json exist)
  python train_and_selfplay.py --mode supervised --lmdb dataset.lmdb --vocab vocab.json --out-dir runs/exp1

  # run self-play and periodically train from replay buffer
  python train_and_selfplay.py --mode selfplay --lmdb dataset.lmdb --vocab vocab.json --out-dir runs/exp1

Read code comments for more tuning guidance.
"""

import os
import json
import random
import math
import time
import pickle
from collections import deque, defaultdict
from argparse import ArgumentParser

import lmdb
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------------
# Utilities & config
# -----------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# LMDB-backed Dataset (expects records written by jsonl_to_lmdb.py)
# -----------------------------


class LMDBChessDataset(torch.utils.data.Dataset):
    """
    LMDB dataset that is safe to use with multiple DataLoader workers.
    It computes the dataset length once (in main process) and reopens
    the LMDB env in each worker after forking.
    """

    def __init__(self, lmdb_path):
        # store path; DO NOT open env here for long-lived use in parent process
        self.lmdb_path = lmdb_path
        self.env = None

        # compute length safely by opening a temporary env and closing it
        tmp_env = lmdb.open(
            self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )
        with tmp_env.begin() as txn:
            # count keys (fast enough for reasonable DB sizes)
            cnt = 0
            cursor = txn.cursor()
            for _ in cursor:
                cnt += 1
            self.length = cnt
        tmp_env.close()

    def __getstate__(self):
        """
        Called when pickling the object to send to worker processes.
        Remove the `env` from state so file descriptors / mmap aren't shared.
        """
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        """
        Called when the dataset is unpickled inside each worker process.
        Reopen the LMDB env here in the worker (safe — opened after fork).
        """
        self.__dict__.update(state)
        self.env = lmdb.open(
            self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # double-check env is open (in case dataset was created on main and not unpickled)
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        key = f"{idx:08d}".encode("ascii")
        with self.env.begin() as txn:
            v = txn.get(key)
            if v is None:
                raise IndexError(idx)
            record = pickle.loads(v)

        board = record["board"]  # numpy (18,8,8)
        board_t = torch.from_numpy(board).to(dtype=torch.float32)
        # policy handling same as before
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
# ResNet model (trunk + two heads)
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
        out += x
        out = self.relu(out)
        return out


class ChessResNet(nn.Module):
    def __init__(self, in_channels, num_blocks=8, channels=128, action_size=4096):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_in = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(
            channels, 32, kernel_size=1
        )  # reduce to small planes
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)

        # value head
        self.value_conv = nn.Conv2d(channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (B, C, 8, 8)
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = self.relu(out)
        for b in self.blocks:
            out = b(out)
        # policy
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = self.relu(p)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # value
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
# Replay buffer for self-play examples
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push_many(self, examples):
        # examples: iterable of (board_np, policy_probs_np, value_scalar)
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
# MCTS (single-threaded for simplicity)
# -----------------------------
class MCTSNode:
    def __init__(self, prior=0.0):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.children = {}  # move_uci -> MCTSNode

    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N


class MCTS:
    def __init__(self, net, vocab, device, cpuct=1.0):
        self.net = net
        self.vocab = vocab  # move -> idx
        self.idx_to_move = {v: k for k, v in vocab.items()}
        self.device = device
        self.cpuct = cpuct

    def _get_priors_and_value(self, board):
        # produce prior probabilities over legal moves and value
        # map network's output to legal moves using vocab; unknown moves get small eps
        board_planes = encode_board(board.fen())  # numpy (18,8,8)
        x = torch.from_numpy(board_planes).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
            logits = logits.squeeze(0).cpu().numpy()  # (action_size,)
            probs = np.exp(logits - np.max(logits))
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
                p = 1e-6
            priors[u] = p
            total += p
        # normalize
        if total > 0:
            for k in priors:
                priors[k] /= total
        else:
            # fallback uniform
            kcnt = len(priors)
            for k in priors:
                priors[k] = 1.0 / kcnt
        return priors, value

    def run(self, root_board, num_sims=80, dirichlet_alpha=0.03, epsilon=0.25):
        root = MCTSNode()
        priors, _ = self._get_priors_and_value(root_board)
        # initialize children with priors
        for mv, p in priors.items():
            root.children[mv] = MCTSNode(prior=p)
        # add dirichlet noise to root priors for exploration
        legal_moves = list(root.children.keys())
        if len(legal_moves) > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
            for i, mv in enumerate(legal_moves):
                root.children[mv].prior = (
                    root.children[mv].prior * (1 - epsilon) + noise[i] * epsilon
                )

        for sim in range(num_sims):
            node = root
            board = root_board.copy()
            path = []  # (node, move_uci)
            # selection
            while True:
                if len(node.children) == 0:
                    break
                # pick best child by PUCT
                best_score = -float("inf")
                best_move = None
                best_child = None
                sqrt_sum = math.sqrt(sum(c.N for c in node.children.values()) + 1)
                for mv, child in node.children.items():
                    U = self.cpuct * child.prior * sqrt_sum / (1 + child.N)
                    Q = child.Q()
                    score = Q + U
                    if score > best_score:
                        best_score = score
                        best_move = mv
                        best_child = child
                # step
                if best_move is None:
                    break
                board.push(chess.Move.from_uci(best_move))
                path.append((node, best_move))
                node = best_child
                if node.N == 0:
                    break
            # expansion & evaluation
            if board.is_game_over():
                # terminal node
                result = board.result()
                if result == "1-0":
                    value = 1.0
                elif result == "0-1":
                    value = -1.0
                else:
                    value = 0.0
            else:
                priors, value = self._get_priors_and_value(board)
                # attach children to leaf
                for mv, p in priors.items():
                    if mv not in node.children:
                        node.children[mv] = MCTSNode(prior=p)
            # backpropagate
            # value is from perspective of the current player to move
            for parent, mv in reversed(path):
                child = parent.children[mv]
                child.N += 1
                child.W += value
                # switch perspective
                value = -value
        # after sims, collect visit counts
        visits = {mv: child.N for mv, child in root.children.items()}
        # produce policy vector over legal moves (in vocab order)
        # convert visits to probs
        total_visits = sum(visits.values())
        policy_probs = {}
        if total_visits > 0:
            for mv, n in visits.items():
                policy_probs[mv] = n / total_visits
        else:
            # uniform
            k = len(visits)
            for mv in visits:
                policy_probs[mv] = 1.0 / k
        return policy_probs


# -----------------------------
# Helpers
# -----------------------------


def encode_board(fen):
    # re-implement minimal encoder here to avoid import loop
    b = chess.Board(fen)
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    piece_map = b.piece_map()
    for sq, piece in piece_map.items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        row = 7 - rank
        col = file
        kind = piece.piece_type
        color = piece.color
        plane_idx = (0 if color else 6) + (kind - 1)
        planes[plane_idx, row, col] = 1.0
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
# Training routines
# -----------------------------


def supervised_train(
    model,
    optimizer,
    scaler,
    dataloader,
    epoch,
    device,
    value_coeff=1.0,
    log_interval=50,
):
    model.train()
    total_loss = 0.0
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    mse_loss = nn.MSELoss()
    pbar = tqdm(
        enumerate(dataloader), total=len(dataloader), desc=f"Train epoch {epoch}"
    )
    for i, (boards, policy, values, played_idx, meta) in pbar:
        # boards: (B, C, 8, 8) OR (C,8,8) if single sample (ensure batch dim)
        if boards.dim() == 3:
            boards = boards.unsqueeze(0)
        boards = boards.to(device)
        bs = boards.size(0)
        # prepare target policy as dense vector
        # policy may be dense tensor or tuple(sparse)
        if isinstance(policy, tuple) or isinstance(policy, list):
            # DataLoader collates tuples to lists -> handle per-sample conversion
            # Build dense target from played_idx
            action_dim = model.policy_fc.out_features
            target_policy = torch.zeros(
                (bs, action_dim), dtype=torch.float32, device=device
            )
            for bi in range(bs):
                p = policy[bi]
                if p is None or (isinstance(p, tuple) and len(p) == 0):
                    if played_idx[bi] >= 0:
                        target_policy[bi, played_idx[bi]] = 1.0
                else:
                    # unlikely pathway; fallback to one-hot
                    if played_idx[bi] >= 0:
                        target_policy[bi, played_idx[bi]] = 1.0
        elif isinstance(policy, torch.Tensor):
            target_policy = policy.to(device)
            if target_policy.dim() == 1:
                target_policy = target_policy.unsqueeze(0)
        else:
            # DataLoader may return None policies; fallback to played_idx
            action_dim = model.policy_fc.out_features
            target_policy = torch.zeros(
                (bs, action_dim), dtype=torch.float32, device=device
            )
            for bi in range(bs):
                if played_idx[bi] >= 0:
                    target_policy[bi, played_idx[bi]] = 1.0
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
    return total_loss / len(dataloader)


# -----------------------------
# Helpers for checkpointing & utils
# -----------------------------


def save_checkpoint(path, model, optimizer, scaler, epoch, extra=None):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("epoch", 0)


# -----------------------------
# Self-play loop
# -----------------------------


def run_selfplay_games(net, vocab, device, num_games=10, sims=80, cpuct=1.0):
    mcts = MCTS(net, vocab, device, cpuct=cpuct)
    examples = []
    for g in range(num_games):
        board = chess.Board()
        game_examples = []
        while not board.is_game_over():
            # run MCTS
            policy_probs = mcts.run(board, num_sims=sims)
            # build dense policy vector in vocab order
            action_dim = len(vocab)
            pol_vec = np.zeros((action_dim,), dtype=np.float32)
            for mv, p in policy_probs.items():
                if mv in vocab:
                    pol_vec[vocab[mv]] = p
            # record position & policy
            game_examples.append((encode_board(board.fen()), pol_vec, None))
            # pick move -- sample using visit probabilities
            moves = list(policy_probs.keys())
            probs = np.array([policy_probs[m] for m in moves], dtype=np.float64)
            probs = probs / (probs.sum() + 1e-12)
            mv_choice = np.random.choice(len(moves), p=probs)
            mv = moves[mv_choice]
            board.push(chess.Move.from_uci(mv))
        # game finished, determine z
        res = board.result()
        if res == "1-0":
            z = 1.0
        elif res == "0-1":
            z = -1.0
        else:
            z = 0.0
        # assign value to examples (from perspective of player to move at that state)
        for bplanes, pol, _ in game_examples:
            examples.append((bplanes, pol, z))
    return examples


# -----------------------------
# Main driver
# -----------------------------


def main():
    ap = ArgumentParser()
    ap.add_argument("--mode", choices=["supervised", "selfplay"], default="supervised")
    ap.add_argument("--lmdb", default="dataset.lmdb")
    ap.add_argument("--vocab", default="vocab.json")
    ap.add_argument("--out-dir", default="runs/exp")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=8)
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

    # build model
    model = ChessResNet(
        in_channels=18,
        num_blocks=args.blocks,
        channels=args.channels,
        action_size=action_size,
    )
    model.to(DEVICE)
    print(
        "Model created. Params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    if args.mode == "supervised":
        dataset = LMDBChessDataset(args.lmdb)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        for epoch in range(1, args.epochs + 1):
            loss = supervised_train(model, optimizer, scaler, loader, epoch, DEVICE)
            print(f"Epoch {epoch} average loss: {loss:.4f}")
            ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, epoch)
            print("Saved checkpoint to", ckpt_path)

    elif args.mode == "selfplay":
        # simple alternating self-play + training
        replay = ReplayBuffer(capacity=args.replay_size)
        # Optionally, seed replay with supervised dataset examples sampled from LMDB to stabilize
        # Here, we do a small seed: sample N random records from dataset
        dataset = LMDBChessDataset(args.lmdb)
        if len(dataset) > 0:
            seed_n = min(20000, len(dataset))
            print(f"Seeding replay buffer with {seed_n} supervised positions...")
            # sample evenly across dataset
            idxs = np.random.choice(len(dataset), seed_n, replace=False)
            for i in tqdm(idxs):
                b, pol, v, played, meta = dataset[i]
                # convert policy to dense numpy array if tensor
                if isinstance(pol, torch.Tensor):
                    pol_np = pol.numpy()
                else:
                    # fallback one-hot on played
                    pol_np = np.zeros((action_size,), dtype=np.float32)
                    if played >= 0:
                        pol_np[played] = 1.0
                replay.push_many([(b.numpy(), pol_np, v)])
            print("Replay seed complete. size=", len(replay))

        # training setup
        batch_size = args.batch_size
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        mse_loss = nn.MSELoss()

        iteration = 0
        while iteration < 1000000:  # run until user stops
            iteration += 1
            print(
                f"Self-play iteration {iteration}: generating {args.selfplay_games} games with {args.mcts_sims} sims each"
            )
            new_examples = run_selfplay_games(
                model, vocab, DEVICE, num_games=args.selfplay_games, sims=args.mcts_sims
            )
            print(f"Generated {len(new_examples)} new positions")
            replay.push_many(new_examples)
            print(f"Replay size: {len(replay)}")

            # train for a small number of steps on replay
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
                    policy_loss = kl_loss(log_probs, policies)
                    value_loss = mse_loss(preds, values)
                    loss = policy_loss + value_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # save checkpoint periodically
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

            # small safety break to avoid infinite loop in examples
            if iteration >= 1000:
                break


if __name__ == "__main__":
    main()
