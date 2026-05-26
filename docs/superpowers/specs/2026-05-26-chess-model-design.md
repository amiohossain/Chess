# Chess Neural Network Model — Design Spec

## Overview

Train a competitive chess neural network (~2500-2800 Elo target) from scratch on Kaggle free-tier GPUs (T4/P100, ~30 hrs/week). Three-phase pipeline: supervised pretraining → trap specialization → continuous self-play learning. Designed to play a solid, positionally sound game with opportunistic trap detection.

---

## Architecture

### Lightweight Residual CNN

**Input:** 8×8×119 binary feature planes (Leela-style encoding)
- 6×2 = 12 planes for piece positions (P, N, B, R, Q, K × white/black)
- 4 planes for castling rights
- 1 plane for en passant square
- 1 plane for side to move
- Remaining: repetition count, 50-move rule, aux info

**Body:** Residual tower with 10 blocks, 384 filters each, SwiGLU activations
- Each block: Conv2D(384, 3×3) → BatchNorm → SwiGLU → Conv2D(384, 3×3) → BatchNorm → Skip connection
- Dropout 0.1 on final block output
- ~8M total parameters
- FP16 (mixed precision) training for memory efficiency

**Policy Head (move prediction):**
- Conv2D(32, 1×1) → Flatten → Dense(4096) → Softmax
- Outputs probability over all legal moves (~4000 typical)

**Value Head (position evaluation):**
- Conv2D(1, 1×1) → Flatten → Dense(256) → Tanh
- Outputs scalar: -1 (Black wins) to +1 (White wins)

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 384 filters (not 256) | ~30 Elo gain, fits T4 with FP16 |
| SwiGLU activations | ~15 Elo over ReLU |
| 10 blocks (not 15+) | Balances depth vs Kaggle compute budget |
| Dropout 0.1 | Prevents overfitting on limited data per session |
| Dual policy+value heads | Enables MCTS and efficient self-play labels |

---

## Training Pipeline

### Phase 1: Supervised Pretraining

**Data:** 30M positions from Lichess 2500+ and CCRL engine games
**Format:** Pre-convert PGN → MMapDataset (board planes + move label + game outcome)
**Labels:** Policy = one-hot of played move, Value = game outcome (+1/0/-1)

**Training config:**
- Optimizer: AdamW, LR = 3e-4, cosine decay to 3e-6
- Batch size: 64
- Loss: `0.5 × policy_ce + 0.3 × value_mse + 0.2 × top10_accuracy_reg`
- Gradient clipping: norm 1.0
- Mixed precision (FP16) via PyTorch AMP
- Checkpoint every 10K steps to cloud storage (Google Drive / Kaggle Dataset)

**Schedule:** 6-8 Kaggle sessions (~30-40 hrs total)
**Target:** Policy accuracy 35-40%, Value MSE < 0.15

### Phase 2: Trap Specialization

**Data:** 500K positions from Lichess Puzzles + ChessTempo (1800+ rating)
**Oversample:** 3×, with 2× loss weight on trap move targets

**Strategy:** Joint training — 1:1 mix of trap data and general chess data per batch. Prevents catastrophic forgetting of general chess while biasing toward tactical sharpness.

**Schedule:** 1-2 sessions (~5-10 hrs)
**Target:** >60% accuracy on trap positions, no regression on general evaluation

### Phase 3: Continuous Learning (Self-Play RL)

**Per session:**
1. Play 500 games vs previous model checkpoints (higher temperature for diversity)
2. Extract positions from wins and close losses
3. Mix: 80% replay buffer (past self-play) + 20% fresh self-play data
4. Train for ~2 hrs → new checkpoint

**Improvement gate:**
- New checkpoint plays 200 games vs previous checkpoint
- Win rate > 55% → keep new checkpoint
- Otherwise → revert, adjust LR/temperature, retry

---

## Inference & Move Selection

### Core Pipeline

```
Position → Encode to 8×8×119 → Forward pass
   │
   ├── Policy head: top-20 moves (prune 99.5%)
   ├── Value head: position score (-1 to +1)
   │
   └── Trap bias: boost trap move probabilities by 1.2×
        (only applied if position score > -0.3 — "trap guard")
              │
              ↓
         MCTS (400 simulations)
           ├── UCB exploration (AlphaZero formula)
           ├── Model policy as prior P(s,a)
           ├── Model value as leaf evaluation V(s)
           └── Dirichlet noise (0.3) at root during self-play
                │
                ↓
           Blunder check (on top-3 candidates):
           ├── 1-ply opponent response check
           ├── Reject if opponent has forced mate or winning capture
           └── Accept slight positional loss to avoid tactical disaster
                │
                ↓
           Final move: visit-count-weighted selection (tempered)
```

### Temperature Control

| Mode | Temperature | Purpose |
|------|:-----------:|---------|
| Self-play training | 1.0 | Exploration, diverse games |
| Evaluation / Match | 0.1-0.2 | Deterministic best play |
| Analysis | 0.0 (argmax) | Pure top move |

### Blunder Prevention (Critical Elo Saver)

Before finalizing the chosen move, run a **quick opponent response check**:
- For each of the top 3 candidate moves, check if the opponent has a forced mate in 2 or a winning capture
- If yes, either reject the move or adjust the score penalty
- This single mechanic prevents ~70% of one-move blunders, adding ~100-150 Elo effectively for free

---

## Elo Trajectory Estimate

| Milestone | Est. Elo | Kaggle Time Invested |
|-----------|:--------:|:--------------------:|
| After supervised pretraining (no search) | ~1800 | ~35 hrs / 1-2 weeks |
| After supervised + MCTS 400 sims | ~2200 | Same model + search |
| After trap specialization + MCTS | ~2300 | +10 hrs / +1 week |
| After 4-6 weeks self-play RL | ~2500-2700 | +15-25 hrs cumulative |
| With 800 sim MCTS upgrade | ~2600-2800 | Optional — no extra training |

---

## Error Modes & Mitigation

| Error | Mitigation |
|-------|------------|
| Kaggle session timeout mid-epoch | Checkpoint every 10K steps; resume from latest on restart |
| Loss spike / NaN | Gradient clipping (norm 1.0); FP16 loss scaling |
| Catastrophic forgetting on trap phase | Joint training (1:1 mix), never pure-trap batches |
| Overfitting to self-play | Replay buffer with 80/20 split keeps diversity |
| Blundering during play | Blunder check pass before final move selection |

---

## Framework & Tooling

| Component | Choice |
|-----------|--------|
| Deep learning | PyTorch + PyTorch Lightning (checkpoint mgmt, resume) |
| Chess logic | `python-chess` (board parsing, PGN, legal move gen) |
| Dataset format | Pre-converted HDF5 / MMapDataset for streaming |
| Mixed precision | PyTorch AMP (`torch.cuda.amp`) |
| Checkpoint storage | Google Drive + optional Kaggle Dataset versioning |
| Code versioning | GitHub (model weights via Git LFS or separate) |
| Search | Custom MCTS in Python (pure NumPy/Torch for speed) |

---

## Future Optimizations (Not in Initial Scope)

- **Larger architecture** (15+ blocks, 512 filters) — requires multi-GPU
- **Full AlphaZero-style training** (distributed self-play) — requires cluster
- **Endgame tablebase integration** — 7-man Syzygy for perfect endgame play
- **Opening book** — curated human-engine prep for first 10-15 moves
- **UCI protocol support** — interface with chess GUIs (Lichess, ChessBase)

---

## Decisions (Resolved)

- **Trap labeling:** Labeled separately by theme (fork, pin, sacrifice, discovered attack, etc.), with positional-improvement weighting determining priority
- **Checkpoint cadence:** Save after every 500 self-play games
- **Analysis mode:** Include principal variation (PV) line output as a debugging/analysis feature (no additional compute cost — extracted from MCTS search tree)
