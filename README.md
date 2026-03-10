# EBL Fundus Classifier
**Assignment 1–3 | IDRiD → APTOS | Baseline vs Energy-Based Learning**

---

## Quick-start (run everything in 3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (see Data Setup below first)
python src/dataset.py --labels data/idrid/labels.csv --image_dir data/idrid/images

# 3. Train baseline, then EBL
python src/train.py --loss softmax    --epochs 10
python src/train.py --loss ebm_margin --margin 1.0 --epochs 10

# 4. Evaluate a checkpoint
python src/eval.py --ckpt runs/<run_folder>/best.pt --split test
```

---

## Project layout

```
ebl_fundus/
├── data/
│   └── idrid/
│       ├── images/          ← all .jpg/.png fundus images go here
│       ├── labels.csv       ← image_id, label
│       └── splits.csv       ← auto-generated on first run
├── src/
│   ├── utils.py             ← seeding, metrics logging
│   ├── dataset.py           ← data loading, splits, augmentation
│   ├── model.py             ← ResNet-18 backbone
│   ├── losses.py            ← CE baseline + EBL margin losses
│   ├── train.py             ← training loop (all experiments)
│   ├── eval.py              ← evaluation + all plots
│   ├── analyze_energy.py    ← Assignment 2 calibration analysis
│   └── visualize_preprocessing.py  ← preprocessing examples (report)
├── runs/                    ← checkpoints + training history (auto-created)
├── outputs/                 ← plots + metrics.csv (auto-created)
├── requirements.txt
└── README.md
```

---

## Data setup

### Option A — Kaggle (easier)
1. Download from: https://www.kaggle.com/datasets/mariaherrerot/idrid-dataset
2. Place all images in `data/idrid/images/`
3. Create `data/idrid/labels.csv` with columns: `image_id, label`
   - `image_id`: filename **without** extension (e.g., `IDRiD_001`)
   - `label`: DR grade (0–4, or binary if you simplify)

### Option B — Official site
1. Register and download from: https://idrid.grand-challenge.org/
2. Same layout as above.

### Verify your setup
```bash
python src/dataset.py
# Should print split sizes and class distribution
```

---

## Run commands (copy these into your report)

### Assignment 1

```bash
# Baseline (Cross-Entropy with class weights)
python src/train.py --dataset idrid --loss softmax --epochs 10

# EBL sum-margin
python src/train.py --dataset idrid --loss ebm_margin --margin 1.0 --epochs 10

# Evaluate best checkpoint on test set
python src/eval.py --ckpt runs/<run_name>/best.pt --split test

# Preprocessing visualization (for report)
python src/visualize_preprocessing.py --data_dir data/idrid --n 4
```

### Assignment 2

```bash
# EBL hard-margin (more stable)
python src/train.py --loss ebm_margin_hard --margin 1.0 --epochs 10

# Lambda regularization sweep
for lam in 0 0.0001 0.001 0.01; do
    python src/train.py --loss ebm_margin --lambda_reg $lam --epochs 10
done

# Calibration + confidence comparison
python src/analyze_energy.py \
    --baseline_ckpt runs/<softmax_run>/best.pt \
    --ebl_ckpt      runs/<ebm_run>/best.pt
```

---

## Outputs (auto-generated)

| File | When |
|---|---|
| `outputs/metrics.csv` | After every training run |
| `outputs/*_confusion_matrix.png` | After evaluation |
| `outputs/*_loss_curve.png` | After evaluation |
| `outputs/*_energy_gap_hist.png` | EBL runs only |
| `outputs/*_energy_gap_correct_vs_wrong.png` | EBL runs only |
| `outputs/preprocessing_examples.png` | visualize_preprocessing.py |
| `outputs/calibration_accuracy_vs_gap.png` | analyze_energy.py |
| `outputs/softmax_vs_energy_confidence.png` | analyze_energy.py |

---

## Energy-Based Learning — brief intuition

Standard cross-entropy trains the model to output **high probability** for the correct class. EBL instead trains it to assign **low energy** to the correct class and **high energy** to all wrong classes, separated by a margin.

```
E(x, k) = -logits[k]          # energy = negative logit

Loss for (x, y):
  sum over k≠y:  max(0,  m  +  E(x,y)  -  E(x,k))
               = max(0,  m  -  logits[y]  +  logits[k])
```

**Margin constraint:** `logits[y] > logits[k] + m` for all wrong classes k.  
When satisfied, the gradient is zero (the constraint is "dead"). Monitor `active_fraction` per batch — printed during training.

**Reference:** LeCun et al., "A Tutorial on Energy-Based Learning" (2006)  
https://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf

---

## Known pitfalls and red flags

| Symptom | Likely cause | Fix |
|---|---|---|
| Macro-F1 < 0.25 after epoch 5 | Class imbalance not handled | Check class weights are loaded |
| Active fraction drops to 0 fast | Margin m too small | Try m = 2.0 or 5.0 |
| Loss explodes (NaN) | Large gradients | Gradient clipping is on (max_norm=1.0); also try smaller LR |
| All predictions = same class | LR too high or no class weights | Use --lr 1e-4, check weighted CE |
| Val F1 flatlines from epoch 1 | Wrong split or data not loading | Re-run dataset.py and inspect loader |

---

## Reproducibility

All experiments fix:
- `random.seed(seed)`
- `numpy.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.backends.cudnn.deterministic = True`

Default seed: `42`. Change with `--seed <N>`.

---

## References

1. LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006).  
   *A Tutorial on Energy-Based Models.*  
   https://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf

2. Porwal, P., et al. (2018). *Indian Diabetic Retinopathy Image Dataset (IDRiD).*  
   https://idrid.grand-challenge.org/

3. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
