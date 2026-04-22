# `training/` — model training scripts

Three training scripts, one per model, plus shared datasets and utils.
All are run directly from the project root (no packaging):

```bash
PYTHONPATH=. python training/train_pose.py          --data datasets/pose    --epochs 30
PYTHONPATH=. python training/train_siamese.py       --data datasets/siamese --epochs 40 --p 8 --k 4
PYTHONPATH=. python training/train_segmentation.py  --data car_parts.yaml   --epochs 100 --export-onnx
```

Outputs (weights + config JSON + TensorBoard logs) land in `models/` by
default. YOLO drops its runs into `car_fraud_poc/<run-name>/`.

## `datasets.py`

| Class | Purpose |
|---|---|
| `PoseDataset` | ImageFolder-style dataset. No horizontal flip in augmentations (it would invert pose labels). |
| `TripletDataset` | Per-vehicle-identity folders; exposes anchor/positive/negative and the identity id for batch-hard mining. |

## `utils.py`

| Helper | Purpose |
|---|---|
| `set_deterministic(seed)` | Seeds Python / numpy / torch / CUDA. |
| `save_checkpoint`, `load_checkpoint` | Atomic checkpoint I/O. |
| `EarlyStopper` | Patience-based stopping (min or max mode). |
| `batch_hard_triplet_loss` | **The important one.** Mines the hardest positive/negative in each batch. |
| `pair_verification_accuracy` | Same/different accuracy at a given cosine threshold — the metric the pipeline actually uses. |
| `cosine_similarity_matrix` | Batched cosine similarity helper. |

## Why batch-hard mining

The original `train_siamese.py` took whatever triplets the DataLoader
handed it and passed them through `nn.TripletMarginLoss`. Random
negatives become trivial to separate after a few epochs — loss drops
quickly, but the embedding space doesn't get much better because the
model is only being pushed on easy examples.

**Batch-hard mining** constructs each batch as P identities × K images
each, then for every anchor selects:

* the **farthest same-identity** embedding as positive (hardest positive)
* the **closest different-identity** embedding as negative (hardest negative)

This forces gradients to flow through the genuinely difficult cases
every step. Combined with the PK sampler, it's the standard modern
recipe for deep-metric learning on faces, cars, and windshields.
