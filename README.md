# Chess Neural Network Model

A competitive chess neural network (~2500-2800 Elo target) trained from scratch on Kaggle free-tier GPUs.

## Pipeline

1. **Phase 1 — Supervised Pretraining:** Train on 30M positions from master-level games
2. **Phase 2 — Trap Specialization:** Fine-tune on 500K tactical positions with theme labels
3. **Phase 3 — Self-Play Learning:** Continuous improvement via self-play games

## Setup

```bash
pip install -r requirements.txt
```
