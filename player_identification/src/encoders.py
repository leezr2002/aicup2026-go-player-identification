"""Replay SGF, fit CosFace encoders and encode games using the competition E4 architecture."""
import argparse, csv, json, math, os, queue, re, threading, time
from pathlib import Path
import numpy as np
from common import HERE, ARTIFACTS, TEST as TEST_DIR
from gallery import read_pool
WORK = HERE / 'outputs/encoder_training'
TRAIN_DIR = HERE.parent / 'data/official/training'
K, PACKED = 64, 181
_AXES = {}
MOVE = re.compile(r";([BW])\[([a-s]{2}|tt|)\]")
HD = re.compile(r"HD\[(\d+)\]")



log = print
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    os.replace(tmp, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def _neighbours():
    table = []
    for p in range(361):
        y, x = divmod(p, 19)
        table.append(tuple(q for q, ok in ((p - 19, y > 0), (p + 19, y < 18), (p - 1, x > 0), (p + 1, x < 18)) if ok))
    return tuple(table)

def parse_moves(sgf):
    moves = []
    for colour, xy in MOVE.findall(sgf):
        c = 1 if colour == "B" else 2
        p = (ord(xy[1]) - 97) * 19 + (ord(xy[0]) - 97) if len(xy) == 2 and xy != "tt" else -1
        moves.append((c, p))
    return moves

def _dead_group(board, start):
    """Return the stones of the group at start if it has no liberty, else None."""
    colour = board[start]
    stack, seen = [start], {start}
    while stack:
        p = stack.pop()
        for q in NEI[p]:
            v = board[q]
            if v == 0:
                return None
            if v == colour and q not in seen:
                seen.add(q)
                stack.append(q)
    return seen

def replay(sgf, colour):
    """Return (K, 181) packed snapshots and the target move count.

    Cell value: bits 0-1 board colour (0 empty / 1 black / 2 white) BEFORE the target
    move, bit 2 = the target move's point. Row = SGF y (top first), col = SGF x.
    """
    target = 1 if colour == "B" else 2
    moves = parse_moves(sgf)
    n = sum(1 for c, _ in moves if c == target)
    out = np.zeros((K, 362), dtype=np.uint8)
    if n == 0:
        return out[:, :PACKED].copy(), 0
    wanted = [1 + (i * (n - 1)) // (K - 1) for i in range(K)]
    need = set(wanted)
    captured = {}
    board = bytearray(361)
    ordinal = 0
    for c, p in moves:
        if c == target:
            ordinal += 1
            if ordinal in need:
                snap = np.frombuffer(bytes(board), dtype=np.uint8).copy()
                if p >= 0:
                    snap[p] |= 4
                captured[ordinal] = snap
        if p < 0 or board[p] != 0:
            continue
        board[p] = c
        opponent = 3 - c
        for q in NEI[p]:
            if board[q] == opponent:
                dead = _dead_group(board, q)
                if dead:
                    for s in dead:
                        board[s] = 0
        dead = _dead_group(board, p)
        if dead:                                       # suicide (rare): remove own group
            for s in dead:
                board[s] = 0
    cells = np.zeros((K, 362), dtype=np.uint8)
    cells[:, :361] = np.stack([captured[w] for w in wanted])
    packed = cells[:, 0::2] | (cells[:, 1::2] << 4)
    return packed, n

def _train_task(args):
    filename, global_start, local_start, count, out_path = args
    csv.field_size_limit(1 << 26)
    mm = np.load(out_path, mmap_mode="r+")
    meta = []
    with (TRAIN_DIR / filename).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        for i, row in enumerate(reader):
            if i < local_start:
                continue
            if i >= local_start + count:
                break
            sgf = row[col["sgf_content"]]
            packed, n = replay(sgf, row[col["color"]])
            mm[global_start + i - local_start] = packed
            hd = HD.search(sgf)
            meta.append((row[col["player_id"]], row[col["game_id"]], row[col["color"]], n,
                         int(hd.group(1)) if hd else -1))
    mm.flush()
    del mm
    return global_start, meta

def torch_setup():
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch

def build_model(n_classes, width, blocks, emb_dim):
    torch = torch_setup()
    from torch import nn
    from torch.nn import functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width),
                                   nn.ReLU(inplace=True), nn.Conv2d(width, width, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(width))
            self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, width // 8, 1), nn.ReLU(inplace=True),
                                    nn.Conv2d(width // 8, width, 1), nn.Sigmoid())

        def forward(self, x):
            y = self.f(x)
            return F.relu(x + y * self.se(y))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3 * K + 3, width, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(width), nn.ReLU(inplace=True))
            self.body = nn.Sequential(*[Block() for _ in range(blocks)])
            self.pool = nn.AdaptiveAvgPool2d(3)
            self.proj = nn.Sequential(nn.Linear(width * 9, emb_dim), nn.BatchNorm1d(emb_dim))
            self.centers = nn.Parameter(torch.randn(n_classes, emb_dim) * 0.01)

        def forward(self, x):
            return self.proj(self.pool(self.body(self.stem(x))).flatten(1))

    return Net()

def expand(torch, packed, colours):
    """packed uint8 (B,K,181) cuda, colours uint8 (B,) 1=B 2=W  ->  bf16 (B,3K+3,19,19)."""
    b = packed.shape[0]
    cells = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)[..., :361].view(b, K, 19, 19)
    stones = cells & 3
    c = colours.view(b, 1, 1, 1)
    planes = torch.stack((stones == c, stones == (3 - c), (cells & 4) != 0), 2).flatten(1, 2)
    dev = packed.device
    if dev not in _AXES:
        axis = torch.linspace(-1, 1, 19, device=dev)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        _AXES[dev] = torch.stack((x, y)).unsqueeze(0)
    side = (3.0 - 2.0 * c.float()).expand(b, 1, 19, 19)
    return torch.cat((planes.to(torch.bfloat16), _AXES[dev].expand(b, -1, -1, -1).to(torch.bfloat16),
                      side.to(torch.bfloat16)), 1).contiguous(memory_format=torch.channels_last)

def encode(model, mm, rows, colours, batch=1024):
    """L2-normalised float32 embeddings for mm[rows]."""
    torch = torch_setup()
    from torch.nn import functional as F
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    out = np.empty((len(rows), model.proj[0].out_features), dtype=np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(rows), batch):
            idx = order[s:s + batch]
            x = torch.from_numpy(np.ascontiguousarray(mm[rows[idx]])).cuda(non_blocking=True)
            c = torch.from_numpy(np.where(np.asarray(colours)[idx] == "B", 1, 2).astype(np.uint8)).cuda()
            out[idx] = F.normalize(model(expand(torch, x, c)).float(), dim=1).cpu().numpy()
    return out

class Loader:
    """Background threads: sample rows (weighted by player), read memmap, pin."""

    def __init__(self, mm, rows, labels, colours, weights, batch, seed, threads=3, depth=6):
        self.mm, self.rows, self.labels, self.colours, self.batch = mm, rows, labels, colours, batch
        self.cdf = np.cumsum(weights / weights.sum())
        self.q = queue.Queue(depth)
        self.stop = False
        self.threads = [threading.Thread(target=self._run, args=(seed + i,), daemon=True) for i in range(threads)]
        for t in self.threads:
            t.start()

    def _run(self, seed):
        import torch
        rng = np.random.default_rng(seed)
        while not self.stop:
            pick = np.minimum(np.searchsorted(self.cdf, rng.random(self.batch)), len(self.rows) - 1)
            pick.sort()
            r = self.rows[pick]
            x = torch.from_numpy(np.ascontiguousarray(self.mm[r])).pin_memory()
            c = torch.from_numpy(self.colours[pick]).pin_memory()
            y = torch.from_numpy(self.labels[pick]).pin_memory()
            self.q.put((x, c, y))

    def next(self):
        return self.q.get()

    def close(self):
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

def train(run_dir, rows, players, colours_bw, steps, args, eval_fn=None, eval_at=(), until=None):
    """CosFace player classification on single games; resumable."""
    torch = torch_setup()
    from torch.nn import functional as F
    run_dir.mkdir(parents=True, exist_ok=True)
    mm = np.load(WORK / "cache/train_seq64.npy", mmap_mode="r")
    codes, labels = np.unique(players, return_inverse=True)
    counts = np.bincount(labels)
    weights = counts[labels].astype(np.float64) ** -0.5          # player weight ~ sqrt(#games)
    colours = np.where(colours_bw == "B", 1, 2).astype(np.uint8)
    torch.manual_seed(args.seed)
    model = build_model(len(codes), args.width, args.blocks, args.emb).cuda().to(memory_format=torch.channels_last)
    body = [p for n, p in model.named_parameters() if n != "centers"]
    opt = torch.optim.AdamW([{"params": body, "weight_decay": args.wd},
                             {"params": [model.centers], "weight_decay": 0.0}], lr=args.lr)
    warm = min(1000, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    start = 0
    last = run_dir / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cuda", weights_only=False)
        if state["steps"] != steps or state["n_classes"] != len(codes):
            raise ValueError(f"{last} was made with different settings; delete the run folder to restart")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start = state["step"]
        log(f"[train] resume {run_dir.name} at step {start}")
    loader = Loader(mm, rows, labels.astype(np.int64), colours, weights, args.batch, args.seed * 1000 + start)
    history = read_json(run_dir / "history.json") if (run_dir / "history.json").exists() else []
    t0, seen = time.time(), 0
    model.train()
    try:
        for step in range(start, until or steps):
            x, c, y = loader.next()
            x, c, y = x.cuda(non_blocking=True), c.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = model(expand(torch, x, c))
            cos = F.normalize(emb.float(), dim=1) @ F.normalize(model.centers.float(), dim=1).T
            logits = args.scale * (cos - args.margin * F.one_hot(y, cos.shape[1]))
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            seen += len(y)
            if (step + 1) % 200 == 0:
                acc = (cos.argmax(1) == y).float().mean().item()
                rate = seen / (time.time() - t0)
                log(f"[train] {run_dir.name} step {step + 1}/{steps} loss {loss.item():.3f} acc {acc:.3f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {rate:.0f} games/s")
                history.append({"step": step + 1, "loss": loss.item(), "acc": acc})
            if (step + 1) % 1000 == 0 or step + 1 == steps or (step + 1) in eval_at:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step + 1, "steps": steps, "n_classes": len(codes)}, last)
                write_json(run_dir / "history.json", history)
            if (step + 1) in eval_at:
                torch.save({"model": model.state_dict(), "n_classes": len(codes)}, run_dir / f"step_{step + 1}.pt")
                if eval_fn is not None:
                    eval_fn(model, step + 1)
                    model.train()
    finally:
        loader.close()
    torch.save({"model": model.state_dict(), "n_classes": len(codes)}, run_dir / "final.pt")
    write_json(run_dir / "done.json", {"steps": steps, "players": int(len(codes)), "rows": int(len(rows))})
    return model

def load_model(path, args):
    torch = torch_setup()
    state = torch.load(path, map_location="cuda", weights_only=False)
    model = build_model(state["n_classes"], args.width, args.blocks, args.emb).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(state["model"])
    model.eval()
    return model

NEI = _neighbours()
