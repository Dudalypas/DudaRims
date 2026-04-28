#!/usr/bin/env python3
"""
Thesis-safe ablation experiment for embedding retrieval training pipeline.

This script implements a limited one-factor-at-a-time ablation study to provide
empirical evidence that important training and retrieval parameters were evaluated.

Key design decisions (thesis-safe):
- Baseline + 7 specific ablation variants (not exhaustive grid search).
- One-factor-at-a-time: each variant changes ONE parameter from baseline.
- Reuses existing retrieval_experiment_cv.py (no new training code).
- Clearly documents which parameters are architecture choices vs evaluated variants.
- Does NOT claim global optimization (instead: "best among tested configurations").
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_ablation_v1"
DEFAULT_DATASET_ROOT = Path(r"C:\Users\vilja\Desktop\PROD_cropped")
DEFAULT_FOLDS_CSV = PROJECT_ROOT / "trained_cropped_classifier" / "retrieval_experiment_v2" / "folds" / "dev_with_folds.csv"

# Baseline configuration: matches verified defaults from retrieval_experiment_cv.py
BASELINE_CONFIG = {
    "name": "baseline",
    "description": "Baseline: embedding_dim=256, lr=1e-3, triplet_margin=0.2, dropout=0.2",
    "img_size": 224,
    "batch_size": 32,
    "epochs": 32,
    "seed": 42,
    "embedding_dim": 256,
    "learning_rate": 1e-3,
    "triplet_margin": 0.2,
    "triplet_weight": 0.5,
    "ce_weight": 1.0,
    "dropout": 0.2,
    "reference_mode": "limited",
    "max_refs_per_class": 20,
    "retrieval_mode": "centroid",
    "topn": 1,
}

# One-factor-at-a-time ablation variants (each changes ONE parameter from baseline)
ABLATION_CONFIGS = [
    {
        "name": "embedding_dim_128",
        "description": "Ablation: embedding_dim=128 (smaller)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "embedding_dim": 128,
    },
    {
        "name": "embedding_dim_512",
        "description": "Ablation: embedding_dim=512 (larger)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "embedding_dim": 512,
    },
    {
        "name": "learning_rate_3e4",
        "description": "Ablation: learning_rate=3e-4 (slower)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "learning_rate": 3e-4,
    },
    {
        "name": "triplet_margin_03",
        "description": "Ablation: triplet_margin=0.3 (larger margin)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "triplet_margin": 0.3,
    },
    {
        "name": "dropout_03",
        "description": "Ablation: dropout=0.3 (more regularization)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "dropout": 0.3,
    },
    {
        "name": "batch_size_16",
        "description": "Ablation: batch_size=16 (smaller batches, sensitivity check)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "batch_size": 16,
    },
    {
        "name": "img_size_192",
        "description": "Ablation: img_size=192 (smaller resolution, sensitivity check)",
        **{k: v for k, v in BASELINE_CONFIG.items() if k != "name" and k != "description"},
        "img_size": 192,
    },
]

# Quick mode: only baseline + 2 key variants (for smoke testing)
QUICK_MODE_CONFIGS = [c for c in [BASELINE_CONFIG] + ABLATION_CONFIGS if c["name"] in ["baseline", "embedding_dim_128", "learning_rate_3e4"]]


@dataclass
class ExperimentResult:
    name: str
    description: str
    config: dict[str, Any]
    output_dir: Path
    cv_summary_path: Path | None
    recall_at_1_mean: float | None
    recall_at_1_std: float | None
    recall_at_3_mean: float | None
    recall_at_3_std: float | None
    recall_at_5_mean: float | None
    recall_at_5_std: float | None
    success: bool
    error_msg: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation experiment for embedding retrieval training. "
            "Runs one-factor-at-a-time variants of the baseline configuration and collects evidence "
            "that important parameters were empirically evaluated."
        )
    )
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--folds-csv", type=str, default=str(DEFAULT_FOLDS_CSV))
    parser.add_argument("--epochs", type=int, default=32, help="Max training epochs for all runs (can be overridden with --quick)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only run baseline, embedding_dim_128, and learning_rate_3e4 (with reduced epochs/folds).",
    )
    parser.add_argument("--quick-epochs", type=int, default=2, help="Epochs for quick mode smoke test.")
    parser.add_argument("--quick-folds", type=int, default=2, help="Number of folds for quick mode (will pick first N folds).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but do not execute.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose output dir already exists.")
    return parser.parse_args()


def extract_cv_metrics(cv_summary_path: Path) -> dict[str, float | None]:
    """Extract centroid retrieval mode metrics from cv_summary.json."""
    if not cv_summary_path.exists():
        return {
            "recall_at_1_mean": None,
            "recall_at_1_std": None,
            "recall_at_3_mean": None,
            "recall_at_3_std": None,
            "recall_at_5_mean": None,
            "recall_at_5_std": None,
        }

    try:
        payload = json.loads(cv_summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error reading {cv_summary_path}: {e}")
        return {
            "recall_at_1_mean": None,
            "recall_at_1_std": None,
            "recall_at_3_mean": None,
            "recall_at_3_std": None,
            "recall_at_5_mean": None,
            "recall_at_5_std": None,
        }

    # Extract metrics for centroid retrieval mode from aggregate section.
    aggregate = payload.get("aggregate", [])
    centroid_row = next((r for r in aggregate if r.get("retrieval_mode") == "centroid"), None)

    if centroid_row:
        return {
            "recall_at_1_mean": float(centroid_row.get("mean_recall@1", 0.0)),
            "recall_at_1_std": float(centroid_row.get("std_recall@1", 0.0)),
            "recall_at_3_mean": float(centroid_row.get("mean_recall@3", 0.0)),
            "recall_at_3_std": float(centroid_row.get("std_recall@3", 0.0)),
            "recall_at_5_mean": float(centroid_row.get("mean_recall@5", 0.0)),
            "recall_at_5_std": float(centroid_row.get("std_recall@5", 0.0)),
        }
    else:
        return {
            "recall_at_1_mean": None,
            "recall_at_1_std": None,
            "recall_at_3_mean": None,
            "recall_at_3_std": None,
            "recall_at_5_mean": None,
            "recall_at_5_std": None,
        }


def run_cv_for_config(
    config: dict[str, Any],
    output_dir: Path,
    folds_csv: str,
    cv_script: Path,
    dry_run: bool = False,
) -> tuple[bool, str | None]:
    """
    Run retrieval_experiment_cv.py with the given config.
    Returns (success, error_msg).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(cv_script),
        "--folds-csv", folds_csv,
        "--output-dir", str(output_dir),
        "--img-size", str(config["img_size"]),
        "--batch-size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]),
        "--seed", str(config["seed"]),
        "--embedding-dim", str(config["embedding_dim"]),
        "--learning-rate", str(config["learning_rate"]),
        "--triplet-margin", str(config["triplet_margin"]),
        "--triplet-weight", str(config["triplet_weight"]),
        "--ce-weight", str(config["ce_weight"]),
        "--dropout", str(config["dropout"]),
        "--reference-mode", str(config["reference_mode"]),
        "--max-refs-per-class", str(config["max_refs_per_class"]),
        "--retrieval-modes", "centroid",  # Only centroid for this ablation.
        "--topn", str(config["topn"]),
    ]

    if dry_run:
        print(f"[DRY-RUN] {' '.join(cmd)}")
        return True, None

    print(f"\n{'='*80}")
    print(f"Running: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        if result.returncode == 0:
            print(f"✓ {config['name']} completed successfully.")
            return True, None
        else:
            err = result.stderr if result.stderr else result.stdout
            msg = f"CV run failed with return code {result.returncode}: {err[:200]}"
            print(f"✗ {config['name']} failed: {msg}")
            return False, msg
    except subprocess.TimeoutExpired:
        msg = f"CV run timed out after 1 hour."
        print(f"✗ {config['name']} timed out.")
        return False, msg
    except Exception as e:
        msg = f"Exception during CV run: {e}"
        print(f"✗ {config['name']} exception: {msg}")
        return False, msg


def select_best_config(results: list[ExperimentResult]) -> tuple[ExperimentResult | None, str]:
    """
    Select best config by: recall@1 mean (primary), then recall@3 mean, then simpler model, then baseline.
    Returns (best_result, justification_text).
    """
    successful = [r for r in results if r.success and r.recall_at_1_mean is not None]
    if not successful:
        return None, "No successful runs."

    # Sort by recall@1 mean (descending), then recall@3 mean, then prefer smaller embedding_dim, then baseline.
    def sort_key(r: ExperimentResult) -> tuple:
        recall1 = r.recall_at_1_mean or -1.0
        recall3 = r.recall_at_3_mean or -1.0
        embedding_dim = r.config["embedding_dim"]
        is_baseline = 1 if r.name == "baseline" else 0
        return (-recall1, -recall3, embedding_dim, -is_baseline)

    successful.sort(key=sort_key)
    best = successful[0]

    justification = (
        f"Selected '{best.name}' based on: "
        f"recall@1={best.recall_at_1_mean:.4f}±{best.recall_at_1_std:.4f}, "
        f"recall@3={best.recall_at_3_mean:.4f}±{best.recall_at_3_std:.4f}. "
    )

    if best.recall_at_1_mean == (successful[1].recall_at_1_mean if len(successful) > 1 else None):
        justification += "(Tied on recall@1; used recall@3 and model complexity as tie-breaker.)"

    return best, justification


def write_results(
    results: list[ExperimentResult],
    output_root: Path,
) -> None:
    """Write results to CSV, JSON, best_config.json, and parameter_justification.md."""
    output_root.mkdir(parents=True, exist_ok=True)

    # CSV results
    csv_path = output_root / "ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "config_name",
            "description",
            "embedding_dim",
            "learning_rate",
            "triplet_margin",
            "dropout",
            "batch_size",
            "img_size",
            "success",
            "recall@1_mean",
            "recall@1_std",
            "recall@3_mean",
            "recall@3_std",
            "recall@5_mean",
            "recall@5_std",
            "error_msg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "config_name": r.name,
                "description": r.description,
                "embedding_dim": r.config["embedding_dim"],
                "learning_rate": r.config["learning_rate"],
                "triplet_margin": r.config["triplet_margin"],
                "dropout": r.config["dropout"],
                "batch_size": r.config["batch_size"],
                "img_size": r.config["img_size"],
                "success": int(r.success),
                "recall@1_mean": r.recall_at_1_mean,
                "recall@1_std": r.recall_at_1_std,
                "recall@3_mean": r.recall_at_3_mean,
                "recall@3_std": r.recall_at_3_std,
                "recall@5_mean": r.recall_at_5_mean,
                "recall@5_std": r.recall_at_5_std,
                "error_msg": r.error_msg or "",
            })
    print(f"Wrote ablation results to: {csv_path}")

    # JSON results
    json_path = output_root / "ablation_results.json"
    json_data = {
        "experiment": "retrieval_ablation_v1",
        "date": str(Path(__file__).resolve().stat().st_mtime),
        "results": [
            {
                "name": r.name,
                "description": r.description,
                "config": r.config,
                "success": r.success,
                "metrics": {
                    "recall@1": {"mean": r.recall_at_1_mean, "std": r.recall_at_1_std},
                    "recall@3": {"mean": r.recall_at_3_mean, "std": r.recall_at_3_std},
                    "recall@5": {"mean": r.recall_at_5_mean, "std": r.recall_at_5_std},
                },
                "error_msg": r.error_msg,
                "cv_summary_path": str(r.cv_summary_path) if r.cv_summary_path else None,
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"Wrote JSON results to: {json_path}")

    # Best config
    best_result, best_justification = select_best_config(results)
    if best_result:
        best_config_path = output_root / "best_config.json"
        best_config_data = {
            "selected_config": best_result.name,
            "justification": best_justification,
            "metrics": {
                "recall@1": {"mean": best_result.recall_at_1_mean, "std": best_result.recall_at_1_std},
                "recall@3": {"mean": best_result.recall_at_3_mean, "std": best_result.recall_at_3_std},
                "recall@5": {"mean": best_result.recall_at_5_mean, "std": best_result.recall_at_5_std},
            },
            "config": best_result.config,
        }
        best_config_path.write_text(json.dumps(best_config_data, indent=2), encoding="utf-8")
        print(f"Wrote best config to: {best_config_path}")


def write_parameter_justification(
    results: list[ExperimentResult],
    output_root: Path,
) -> None:
    """Write thesis-safe parameter justification markdown with Lithuanian wording."""
    md_path = output_root / "parameter_justification.md"

    best_result, best_justification = select_best_config(results)

    # Build results table
    results_table = "| Config | Recall@1 (mean±std) | Recall@3 (mean±std) | Recall@5 (mean±std) | Status |\n"
    results_table += "|--------|-------------------|-------------------|-------------------|--------|\n"
    for r in results:
        if r.success and r.recall_at_1_mean is not None:
            r1 = f"{r.recall_at_1_mean:.4f}±{r.recall_at_1_std:.4f}"
            r3 = f"{r.recall_at_3_mean:.4f}±{r.recall_at_3_std:.4f}"
            r5 = f"{r.recall_at_5_mean:.4f}±{r.recall_at_5_std:.4f}"
            status = "✓ Best" if (best_result and r.name == best_result.name) else "✓ OK"
        else:
            r1 = r3 = r5 = "—"
            status = f"✗ Failed" if r.error_msg else "✗ Skipped"
        results_table += f"| {r.name} | {r1} | {r3} | {r5} | {status} |\n"

    markdown_content = f"""# Embedding Retrieval Model: Ablation Study Evidence

## 1. Experiment Goal

This document presents evidence that important training and retrieval parameters for the embedding retrieval model were evaluated empirically through a limited one-factor-at-a-time ablation study.

**Key clarification:** This study does NOT claim global optimization across all possible parameter combinations. Instead, it provides evidence that selected parameters were empirically tested and the best performing configuration among those variants was identified.

## 2. One-Factor-At-A-Time Ablation Design

Each ablation variant changes exactly ONE parameter from a verified baseline configuration. This approach:
- Isolates the effect of individual parameters
- Remains computationally feasible for thesis timelines
- Provides clear evidence of empirical parameter evaluation
- Maintains clarity about which choices are architectural vs. empirically optimized

### Baseline Configuration
```json
{{
  "img_size": 224,
  "batch_size": 32,
  "epochs": 32,
  "seed": 42,
  "embedding_dim": 256,
  "learning_rate": 1e-3,
  "triplet_margin": 0.2,
  "triplet_weight": 0.5,
  "ce_weight": 1.0,
  "dropout": 0.2,
  "retrieval_mode": "centroid"
}}
```

### Ablation Variants
1. **embedding_dim_128**: Tests smaller embedding dimensionality (256 → 128)
2. **embedding_dim_512**: Tests larger embedding dimensionality (256 → 512)
3. **learning_rate_3e4**: Tests slower learning rate (1e-3 → 3e-4)
4. **triplet_margin_03**: Tests larger triplet margin (0.2 → 0.3)
5. **dropout_03**: Tests increased regularization (0.2 → 0.3)
6. **batch_size_16**: Tests smaller batch size (32 → 16); limited sensitivity check
7. **img_size_192**: Tests smaller input resolution (224 → 192); limited sensitivity check

## 3. Important Methodological Clarifications for Thesis

### 3.1 Adam Optimizer ≠ Hyperparameter Optimization

The model training uses **TensorFlow Adam optimizer** with a fixed learning rate parameter. This is NOT a hyperparameter search method.

**Distinction:**
- **Adam:** An adaptive gradient descent optimizer that rescales gradients per parameter independently. Adam's internal mechanics (momentum, adaptive learning rates per weight) do not search over the hyperparameter space; instead, they improve convergence within a fixed learning rate.
- **Hyperparameter search:** Explicit enumeration of candidate values (e.g., trying learning_rate ∈ {{1e-3, 3e-4, 1e-4}}) and selecting the best based on validation performance.

This ablation study provides evidence of the latter (explicit hyperparameter evaluation), not the former.

### 3.2 Seed as Reproducibility Control, NOT Optimization Parameter

**Seed (random_seed = 42) is NOT treated as an optimization parameter.**

Why:
- Seed controls initialization randomness and data shuffle order for reproducibility.
- Varying seed to maximize performance would be a form of overfitting to random initialization.
- Thesis-quality work fixes seed to ensure deterministic, reproducible results.
- Different seeds may yield different metrics due to random variance, not fundamental model differences.

**Thesis statement:** "Seed naudotas atkartojamumui užtikrinti, o ne rezultatams optimizuoti. Visos eksperimentinės seros naudojo seed=42."

### 3.3 Epochs as Training Budget (Not Optimized), EarlyStopping as Automatic Best Selection

**Epochs (maximum=32) is NOT treated as an optimized hyperparameter.**

Why:
- Epochs define the maximum training budget (computational/time limit).
- **EarlyStopping callback** monitors validation accuracy and stops training early if no improvement for 6 epochs, with `restore_best_weights=True`.
- This means the effective number of epochs trained is adaptive and determined automatically by validation performance, not a hyperparameter to search over.
- Including epochs in a sweep would conflate training budget with model capacity/learning dynamics.

**Thesis statement:** "Epochų skaičius naudotas kaip maksimali mokymo riba. Geriausios modelio būsenos pasirinktos automatizuotai EarlyStopping ir ModelCheckpoint mechanizmais, kurie stebėjo validacijos tikslumą."

### 3.4 Internal Validation Ratio as Protocol Parameter (Not Optimized)

In final training (retrieval_experiment_final.py), an internal validation split (default 10% of TRAIN+VAL) is used during training for EarlyStopping monitoring. This is NOT an optimization parameter.

Why:
- It is a **protocol choice** for final model training workflow.
- It ensures test set remains untouched during all training/selection.
- Varying this ratio to optimize performance would risk overfitting to the internal validation split.

**Not included in this ablation.**

### 3.5 img_size and batch_size: Limited Sensitivity Checks

Variants **img_size_192** and **batch_size_16** are included as limited sensitivity checks, NOT primary hyperparameter searches.

Why:
- These are typically constrained by hardware (memory, latency requirements).
- Their effect on final performance is less direct than embedding_dim, learning_rate, or loss weights.
- They demonstrate due diligence in testing but are secondary to core embedding model choices.

**Thesis statement:** "img_size ir batch_size pateikiami kaip ribota jautruminė analizė, nuo kurios tiesiogiai priklauso aparatinės įrangos apribojimai ir taikymo reikalavimai, o ne modelio reprezentacinė galia."

## 4. Experimental Results

### Results Summary
{results_table}

### Selected Best Configuration

**Configuration:** `{best_result.name if best_result else 'None'}`

**Justification:** {best_justification if best_result else 'No successful runs.'}

## 5. Validation Methodology

All experiments use **5-fold stratified cross-validation** on the development set (TRAIN+VAL split combined). Each fold:
- Splits development data into training (4 folds) and validation (1 fold).
- Trains a new model from scratch with the variant hyperparameters.
- Monitors `val_logits_top1` accuracy; EarlyStopping stops if no improvement for 6 epochs.
- Evaluates on the held-out validation fold using centroid-based retrieval.
- Records recall@1, recall@3, recall@5 with mean and standard deviation across folds.

This ensures robustness and reduces variance due to random train/val split.

## 6. Limitations and Honest Assessment

1. **Limited search space:** Only 8 configurations tested (1 baseline + 7 one-factor ablations). Full grid search (all combinations) would be computationally prohibitive and is not necessary for thesis evidence.

2. **No probabilistic optimization:** This is not a Bayesian optimization or random search that theoretically explores high-dimensional space. It is targeted ablation of key parameters based on domain knowledge and common practice.

3. **No guarantee of global optimum:** The selected configuration is the best among tested variants, NOT necessarily globally optimal across all possible hyperparameter combinations.

4. **Architecture choices are fixed:** MobileNetV3Small backbone, batch_hard_triplet_loss, L2 normalization, and dual-head (embedding + logits) architecture are treated as fixed architectural choices, not optimized parameters. These reflect state-of-the-art metric learning practices and are justified separately.

5. **Cross-validation variance:** Results have error bars (std across folds) which reflect natural variance in k-fold CV. Statistical significance is not formally tested; instead, results are interpreted with variance context.

## 7. Thesis-Safe Summary (Lithuanian)

Atliktas ribotas abliacijos eksperimentas, kurio tikslas – pateikti empirinį patvirtinimą, kad svarbiausieji mokymo parametrai buvo sistemingai išbandyti.

**Optimalumas** šiame kontekste reiškia geriausią rezultatą tarp patikrintų konfigūracijų, o ne globalią visų kombinacijų paieską.

Pagrindinės parametrinės prielaidos:
- **Seed (=42)** naudotas atkartojamumui, o ne rezultatams optimizuoti.
- **Epochų skaičius (max=32)** apibrėžtas kaip maksimali mokymo riba; geriausios svorio pasirinktos per EarlyStopping.
- **Adam optimizatorius** nėra hiperparametrų paieška, o adaptyvus gradientų metodas su fiksuotu learning rate parametru.
- **Vidinės validacijos santykis (10%)** yra mokymo protokolo pasirinkimas, o ne optimizuojamas parametras.

Konkretus abliacijos testas izoliavo individualių parametrų poveikį: embedding dimensionalumo (128, 256, 512), learning rate (1e-3, 3e-4), triplet margin (0.2, 0.3), dropout (0.2, 0.3), ir ribotą aparatinės jautruminę analizę (batch_size, img_size).

Pilna visų kombinacijų paieška neatlikta dėl skaičiavimo sąnaudų ir laiko apribojimų. Vietoj to, šis metodas pateikia praktinius, patikrinus empirinį patvirtinimą, kuris yra pakankamas moksliniam darbui.

## 8. Recommended Thesis Wording

**English:**
"To provide empirical evidence that important training hyperparameters were evaluated, a one-factor-at-a-time ablation study was conducted. The model was trained under eight configurations: one baseline and seven single-parameter variants. Each variant was evaluated using 5-fold stratified cross-validation. The configuration achieving the highest recall@1 (primary metric) was selected. While this approach does not perform exhaustive grid search across all parameter combinations, it provides clear evidence of systematic empirical parameter evaluation."

**Lithuanian:**
"Norėdami pateikti empirinį patvirtinimą, kad svarbieji mokymo hiperparametrai buvo sistemingai išbandyti, atliktas vieno-faktoriaus-iš-karto abliacijos eksperimentas. Modelis buvo mokytas aštuomis konfigūracijomis: viena bazinė ir septyni vieno parametro variantai. Kiekvienas variantas buvo įvertintas naudojant 5-fold stratifikuotą kryžminio patvirtinimo metodą. Pasirinkta konfigūracija, kuri pasiekė aukščiausią recall@1 (pirminė metrika). Nors šis metodas neatlieka išsamios tinklelio paieškos visose parametrų kombinacijose, jis pateikia aiškų sisteminės empirinės parametrų įvertinimo patvirtinimą."

---

*Ablation study completed: {len([r for r in results if r.success])} / {len(results)} configurations successful.*
"""

    md_path.write_text(markdown_content, encoding="utf-8")
    print(f"Wrote parameter justification to: {md_path}")


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_dir)
    cv_script = PROJECT_ROOT / "xtools" / "retrieval_experiment_cv.py"

    if not cv_script.exists():
        print(f"Error: {cv_script} not found.")
        sys.exit(1)

    if not Path(args.folds_csv).exists():
        print(f"Error: Folds CSV not found: {args.folds_csv}")
        sys.exit(1)

    # Select configs based on --quick flag
    configs_to_run = QUICK_MODE_CONFIGS if args.quick else [BASELINE_CONFIG] + ABLATION_CONFIGS

    # Override epochs for quick mode
    if args.quick:
        for cfg in configs_to_run:
            cfg["epochs"] = args.quick_epochs

    print(f"\n{'='*80}")
    print(f"Retrieval Ablation Experiment")
    print(f"{'='*80}")
    print(f"Output root: {output_root}")
    print(f"Configurations to run: {len(configs_to_run)}")
    print(f"Dry-run: {args.dry_run}")
    print(f"Quick mode: {args.quick}")
    print(f"{'='*80}\n")

    results: list[ExperimentResult] = []

    for config in configs_to_run:
        config_output_dir = output_root / config["name"]

        # Check if should skip
        if args.skip_existing and config_output_dir.exists():
            print(f"Skipping {config['name']} (output dir exists).")
            cv_summary_path = config_output_dir / "cv_summary.json"
            metrics = extract_cv_metrics(cv_summary_path)
            result = ExperimentResult(
                name=config["name"],
                description=config["description"],
                config=config,
                output_dir=config_output_dir,
                cv_summary_path=cv_summary_path,
                recall_at_1_mean=metrics.get("recall_at_1_mean"),
                recall_at_1_std=metrics.get("recall_at_1_std"),
                recall_at_3_mean=metrics.get("recall_at_3_mean"),
                recall_at_3_std=metrics.get("recall_at_3_std"),
                recall_at_5_mean=metrics.get("recall_at_5_mean"),
                recall_at_5_std=metrics.get("recall_at_5_std"),
                success=True,
            )
            results.append(result)
            continue

        # Run CV for this config
        success, error_msg = run_cv_for_config(
            config=config,
            output_dir=config_output_dir,
            folds_csv=args.folds_csv,
            cv_script=cv_script,
            dry_run=args.dry_run,
        )

        # Extract metrics
        cv_summary_path = config_output_dir / "cv_summary.json"
        metrics = extract_cv_metrics(cv_summary_path)

        result = ExperimentResult(
            name=config["name"],
            description=config["description"],
            config=config,
            output_dir=config_output_dir,
            cv_summary_path=cv_summary_path if success else None,
            recall_at_1_mean=metrics.get("recall_at_1_mean"),
            recall_at_1_std=metrics.get("recall_at_1_std"),
            recall_at_3_mean=metrics.get("recall_at_3_mean"),
            recall_at_3_std=metrics.get("recall_at_3_std"),
            recall_at_5_mean=metrics.get("recall_at_5_mean"),
            recall_at_5_std=metrics.get("recall_at_5_std"),
            success=success,
            error_msg=error_msg,
        )
        results.append(result)

    # Write results and documentation
    if not args.dry_run:
        write_results(results, output_root)
        write_parameter_justification(results, output_root)

        best_result, _ = select_best_config(results)
        if best_result:
            print(f"\n{'='*80}")
            print(f"Best configuration: {best_result.name}")
            print(f"Recall@1: {best_result.recall_at_1_mean:.4f}±{best_result.recall_at_1_std:.4f}")
            print(f"Recall@3: {best_result.recall_at_3_mean:.4f}±{best_result.recall_at_3_std:.4f}")
            print(f"Recall@5: {best_result.recall_at_5_mean:.4f}±{best_result.recall_at_5_std:.4f}")
            print(f"{'='*80}\n")

    print("Done.")


if __name__ == "__main__":
    main()
