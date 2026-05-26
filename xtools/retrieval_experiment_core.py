from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif", ".avif"}
TOP_KS = (1, 3, 5)


@dataclass
class ImageRecord:
    path: Path
    label: str
    source_split: str | None = None
    fold: int | None = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def list_class_images(split_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    if not split_dir.exists():
        return records

    class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    for class_dir in class_dirs:
        for p in sorted(class_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                records.append(ImageRecord(path=p, label=class_dir.name, source_split=split_dir.name))
    return records


def combine_dev_records(dataset_root: Path) -> tuple[list[ImageRecord], list[ImageRecord]]:
    train_records = list_class_images(dataset_root / "train")
    val_records = list_class_images(dataset_root / "val")
    test_records = list_class_images(dataset_root / "test")

    dev = train_records + val_records
    if not dev:
        raise RuntimeError(f"No development records found under {dataset_root / 'train'} and {dataset_root / 'val'}")
    if not test_records:
        raise RuntimeError(f"No test records found under {dataset_root / 'test'}")

    return dev, test_records


def stratified_kfold_assign(records: list[ImageRecord], n_folds: int, seed: int) -> list[ImageRecord]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        grouped[r.label].append(r)

    rng = random.Random(seed)
    out: list[ImageRecord] = []

    for label in sorted(grouped):
        rows = grouped[label][:]
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            out.append(ImageRecord(path=row.path, label=row.label, source_split=row.source_split, fold=i % n_folds))

    return out


def class_distribution(records: list[ImageRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.label] += 1
    return dict(sorted(counts.items()))


def fold_distribution(records: list[ImageRecord], n_folds: int) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {i: {} for i in range(n_folds)}
    for i in range(n_folds):
        fold_rows = [r for r in records if r.fold == i]
        out[i] = class_distribution(fold_rows)
    return out


def write_records_csv(records: list[ImageRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source_split", "fold"])
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "path": str(r.path),
                    "label": r.label,
                    "source_split": r.source_split or "",
                    "fold": "" if r.fold is None else r.fold,
                }
            )


def read_records_csv(path: Path) -> list[ImageRecord]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[ImageRecord] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold_text = (row.get("fold") or "").strip()
            rows.append(
                ImageRecord(
                    path=Path((row.get("path") or "").strip()),
                    label=(row.get("label") or "").strip(),
                    source_split=(row.get("source_split") or "").strip() or None,
                    fold=int(fold_text) if fold_text else None,
                )
            )
    return rows


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _pil_preprocess(path_str: tf.Tensor | bytes, img_size: int) -> np.ndarray:
    if hasattr(path_str, "numpy"):
        raw = path_str.numpy()
    else:
        raw = path_str

    if isinstance(raw, np.ndarray):
        raw = raw.item()

    if isinstance(raw, (bytes, bytearray)):
        p = Path(raw.decode("utf-8"))
    else:
        p = Path(str(raw))

    with Image.open(p) as img:
        oriented = ImageOps.exif_transpose(img).convert("RGB")
        resized = oriented.resize((img_size, img_size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _tf_load(path: tf.Tensor, label: tf.Tensor, img_size: int) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    image = tf.py_function(
        func=lambda p: _pil_preprocess(p, img_size),
        inp=[path],
        Tout=tf.float32,
    )
    image.set_shape((img_size, img_size, 3))
    cls = tf.cast(label, tf.int32)
    return image, {"embedding": cls, "logits": cls}


def _tf_load_infer(path: tf.Tensor, img_size: int) -> tf.Tensor:
    image = tf.py_function(
        func=lambda p: _pil_preprocess(p, img_size),
        inp=[path],
        Tout=tf.float32,
    )
    image.set_shape((img_size, img_size, 3))
    return image


def build_label_index(records: list[ImageRecord]) -> dict[str, int]:
    classes = sorted({r.label for r in records})
    return {c: i for i, c in enumerate(classes)}


def build_train_val_datasets(
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    label_to_idx: dict[str, int],
    img_size: int,
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_paths = [str(r.path) for r in train_records]
    train_labels = [label_to_idx[r.label] for r in train_records]
    val_paths = [str(r.path) for r in val_records]
    val_labels = [label_to_idx[r.label] for r in val_records]

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(len(train_paths), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(lambda p, y: _tf_load(p, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(lambda p, y: _tf_load(p, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def build_infer_dataset(records: list[ImageRecord], img_size: int, batch_size: int) -> tf.data.Dataset:
    paths = [str(r.path) for r in records]
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: _tf_load_infer(p, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _pairwise_distance(embeddings: tf.Tensor) -> tf.Tensor:
    dot = tf.matmul(embeddings, embeddings, transpose_b=True)
    sq = tf.linalg.diag_part(dot)
    dist = tf.expand_dims(sq, 1) - 2.0 * dot + tf.expand_dims(sq, 0)
    return tf.maximum(dist, 0.0)


def batch_hard_triplet_loss(labels: tf.Tensor, embeddings: tf.Tensor, margin: float = 0.2) -> tf.Tensor:
    labels = tf.reshape(tf.cast(labels, tf.int32), [-1, 1])
    pdist = _pairwise_distance(embeddings)

    same = tf.equal(labels, tf.transpose(labels))
    same = tf.logical_and(same, ~tf.eye(tf.shape(labels)[0], dtype=tf.bool))
    diff = ~tf.equal(labels, tf.transpose(labels))

    pos_dist = tf.where(same, pdist, tf.zeros_like(pdist))
    hardest_pos = tf.reduce_max(pos_dist, axis=1)

    neg_fill = tf.fill(tf.shape(pdist), tf.cast(1e9, pdist.dtype))
    neg_dist = tf.where(diff, pdist, neg_fill)
    hardest_neg = tf.reduce_min(neg_dist, axis=1)

    has_pos = tf.reduce_any(same, axis=1)
    has_neg = tf.reduce_any(diff, axis=1)
    valid = tf.logical_and(has_pos, has_neg)

    losses = tf.maximum(hardest_pos - hardest_neg + margin, 0.0)
    losses = tf.where(valid, losses, tf.zeros_like(losses))

    denom = tf.maximum(tf.reduce_sum(tf.cast(valid, tf.float32)), 1.0)
    return tf.reduce_sum(losses) / denom


def make_triplet_loss(margin: float):
    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return batch_hard_triplet_loss(y_true, y_pred, margin=margin)

    return _loss


def build_retrieval_training_model(
    img_size: int,
    num_classes: int,
    embedding_dim: int,
    learning_rate: float,
    triplet_margin: float,
    triplet_weight: float,
    ce_weight: float,
    dropout: float,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image", dtype=tf.float32)

    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    x = aug(inputs)

    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )

    # Saugiausiai
    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)

    emb_raw = tf.keras.layers.Dense(embedding_dim, activation=None, name="embedding_dense")(x)
    embedding = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name="embedding")(emb_raw)

    logits = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(embedding)

    model = tf.keras.Model(inputs=inputs, outputs={"embedding": embedding, "logits": logits}, name="retrieval_mnv3small")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "embedding": make_triplet_loss(triplet_margin),
            "logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        },
        loss_weights={"embedding": triplet_weight, "logits": ce_weight},
        metrics={
            "logits": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
            ]
        },
    )
    return model


def train_one_run(
    train_records: list[ImageRecord],
    val_records: list[ImageRecord],
    output_dir: Path,
    img_size: int,
    batch_size: int,
    epochs: int,
    seed: int,
    embedding_dim: int,
    learning_rate: float,
    triplet_margin: float,
    triplet_weight: float,
    ce_weight: float,
    dropout: float,
    early_stopping_patience: int = 5,
    restore_best_weights: bool = True,
) -> tuple[tf.keras.Model, dict[str, float], dict[str, int]]:
    if not train_records:
        raise RuntimeError("No train records for this run.")
    if not val_records:
        raise RuntimeError("No validation records for this run.")

    set_global_seed(seed)

    all_records = train_records + val_records
    label_to_idx = build_label_index(all_records)

    train_ds, val_ds = build_train_val_datasets(
        train_records=train_records,
        val_records=val_records,
        label_to_idx=label_to_idx,
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = output_dir / "best.weights.h5"
    history_csv = output_dir / "history.csv"

    model = build_retrieval_training_model(
        img_size=img_size,
        num_classes=len(label_to_idx),
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        triplet_margin=triplet_margin,
        triplet_weight=triplet_weight,
        ce_weight=ce_weight,
        dropout=dropout,
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="val_logits_top1",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_logits_top1",
            mode="max",
            patience=early_stopping_patience,
            restore_best_weights=restore_best_weights,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(history_csv), append=False),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Reloadina geriausia checkpointa svoriu vertinimui/esksportui
    model.load_weights(best_weights_path)

    val_metrics = model.evaluate(val_ds, return_dict=True, verbose=1)
    val_metrics = {k: float(v) for k, v in val_metrics.items()}

    model.save(output_dir / "best.keras")
    (output_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    return model, val_metrics, {i: idx_to_label[i] for i in sorted(idx_to_label)}


def extract_embedding_model(trained_model: tf.keras.Model) -> tf.keras.Model:
    emb = trained_model.get_layer("embedding").output
    return tf.keras.Model(inputs=trained_model.input, outputs=emb, name=f"{trained_model.name}_embedding")


def infer_embeddings(
    embedding_model: tf.keras.Model,
    records: list[ImageRecord],
    img_size: int,
    batch_size: int,
) -> tuple[np.ndarray, list[str], list[Path]]:
    if not records:
        return np.zeros((0, 0), dtype=np.float32), [], []

    ds = build_infer_dataset(records, img_size=img_size, batch_size=batch_size)
    emb = embedding_model.predict(ds, verbose=0)
    emb = np.asarray(emb, dtype=np.float32)

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    emb = emb / norms

    labels = [r.label for r in records]
    paths = [r.path for r in records]
    return emb, labels, paths


def build_reference_db(
    embeddings: np.ndarray,
    labels: list[str],
    mode: str,
    max_refs_per_class: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for vec, label in zip(embeddings, labels):
        grouped[label].append(vec)

    rng = random.Random(seed)
    db: dict[str, dict[str, Any]] = {}

    for label in sorted(grouped):
        vectors = grouped[label]
        if mode == "limited" and max_refs_per_class > 0 and len(vectors) > max_refs_per_class:
            idx = list(range(len(vectors)))
            rng.shuffle(idx)
            idx = sorted(idx[:max_refs_per_class])
            vectors = [vectors[i] for i in idx]

        stack = np.stack(vectors, axis=0)
        centroid = np.mean(stack, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 1e-12:
            centroid = centroid / c_norm

        db[label] = {
            "vectors": vectors,
            "centroid": centroid.astype(np.float32),
        }

    return db


def _class_scores(
    query_vec: np.ndarray,
    reference_db: dict[str, dict[str, Any]],
    retrieval_mode: str,
    topn: int,
) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []

    for label, payload in reference_db.items():
        if retrieval_mode == "centroid":
            score = float(np.dot(query_vec, payload["centroid"]))
        else:
            sims = [float(np.dot(query_vec, v)) for v in payload["vectors"]]
            if not sims:
                score = -1.0
            elif retrieval_mode == "multi_max":
                score = max(sims)
            elif retrieval_mode == "multi_topn_avg":
                sims.sort(reverse=True)
                n = max(1, min(topn, len(sims)))
                score = float(np.mean(sims[:n]))
            else:
                raise ValueError(f"Unsupported retrieval_mode={retrieval_mode}")
        scores.append((label, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    query_labels: list[str],
    reference_db: dict[str, dict[str, Any]],
    retrieval_mode: str,
    topn: int,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    totals = {k: 0 for k in TOP_KS}
    per_class: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "hits@1": 0, "hits@3": 0, "hits@5": 0})
    prediction_rows: list[dict[str, Any]] = []
    top_k_capture = max(TOP_KS)

    for query_index, (vec, true_label) in enumerate(zip(query_embeddings, query_labels)):
        ranked = _class_scores(vec, reference_db, retrieval_mode=retrieval_mode, topn=topn)
        ranked_labels = [lbl for lbl, _ in ranked]
        top_k_labels = ranked_labels[:top_k_capture]
        top_k_scores = [float(score) for _, score in ranked[:top_k_capture]]

        per_class[true_label]["count"] += 1

        prediction_rows.append(
            {
                "query_index": int(query_index),
                "true_label": true_label,
                "top1_predicted_label": ranked_labels[0] if ranked_labels else "",
                "top_k_labels": json.dumps(top_k_labels, ensure_ascii=False),
                "top_k_scores": json.dumps(top_k_scores),
                "retrieval_mode": retrieval_mode,
                "topn": int(topn),
            }
        )

        for k in TOP_KS:
            hit = 1 if true_label in ranked_labels[:k] else 0
            totals[k] += hit
            per_class[true_label][f"hits@{k}"] += hit

    n = max(1, len(query_labels))
    summary = {
        "recall@1": totals[1] / n,
        "recall@3": totals[3] / n,
        "recall@5": totals[5] / n,
    }

    per_class_rows: list[dict[str, Any]] = []
    for label in sorted(per_class):
        row = per_class[label]
        count = max(1, int(row["count"]))
        per_class_rows.append(
            {
                "label": label,
                "count": int(row["count"]),
                "recall@1": row["hits@1"] / count,
                "recall@3": row["hits@3"] / count,
                "recall@5": row["hits@5"] / count,
            }
        )

    return summary, per_class_rows, prediction_rows


def aggregate_fold_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["retrieval_mode"]), int(row["topn"]))].append(row)

    out: list[dict[str, Any]] = []
    for (mode, topn), items in sorted(grouped.items()):
        r1 = np.asarray([float(i["recall@1"]) for i in items], dtype=np.float64)
        r3 = np.asarray([float(i["recall@3"]) for i in items], dtype=np.float64)
        r5 = np.asarray([float(i["recall@5"]) for i in items], dtype=np.float64)

        out.append(
            {
                "retrieval_mode": mode,
                "topn": topn,
                "folds": len(items),
                "mean_recall@1": float(np.mean(r1)),
                "std_recall@1": float(np.std(r1)),
                "mean_recall@3": float(np.mean(r3)),
                "std_recall@3": float(np.std(r3)),
                "mean_recall@5": float(np.mean(r5)),
                "std_recall@5": float(np.std(r5)),
            }
        )

    return out


def aggregate_classification_fold_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["retrieval_mode"]), int(row["topn"]))].append(row)

    out: list[dict[str, Any]] = []
    for (mode, topn), items in sorted(grouped.items()):
        macro_precision = np.asarray([float(i["macro_precision"]) for i in items], dtype=np.float64)
        macro_recall = np.asarray([float(i["macro_recall"]) for i in items], dtype=np.float64)
        macro_f1 = np.asarray([float(i["macro_f1"]) for i in items], dtype=np.float64)
        weighted_precision = np.asarray([float(i["weighted_precision"]) for i in items], dtype=np.float64)
        weighted_recall = np.asarray([float(i["weighted_recall"]) for i in items], dtype=np.float64)
        weighted_f1 = np.asarray([float(i["weighted_f1"]) for i in items], dtype=np.float64)

        out.append(
            {
                "retrieval_mode": mode,
                "topn": topn,
                "folds": len(items),
                "macro_precision_mean": float(np.mean(macro_precision)),
                "macro_precision_std": float(np.std(macro_precision)),
                "macro_recall_mean": float(np.mean(macro_recall)),
                "macro_recall_std": float(np.std(macro_recall)),
                "macro_f1_mean": float(np.mean(macro_f1)),
                "macro_f1_std": float(np.std(macro_f1)),
                "weighted_precision_mean": float(np.mean(weighted_precision)),
                "weighted_precision_std": float(np.std(weighted_precision)),
                "weighted_recall_mean": float(np.mean(weighted_recall)),
                "weighted_recall_std": float(np.std(weighted_recall)),
                "weighted_f1_mean": float(np.mean(weighted_f1)),
                "weighted_f1_std": float(np.std(weighted_f1)),
            }
        )

    return out


def aggregate_classification_per_class_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["retrieval_mode"]), int(row["topn"]), str(row["label"]))].append(row)

    out: list[dict[str, Any]] = []
    for (mode, topn, label), items in sorted(grouped.items()):
        precision = np.asarray([float(i["precision"]) for i in items], dtype=np.float64)
        recall = np.asarray([float(i["recall"]) for i in items], dtype=np.float64)
        f1 = np.asarray([float(i["f1_score"]) for i in items], dtype=np.float64)
        support = np.asarray([float(i["support"]) for i in items], dtype=np.float64)

        out.append(
            {
                "retrieval_mode": mode,
                "topn": topn,
                "label": label,
                "folds": len(items),
                "precision_mean": float(np.mean(precision)),
                "precision_std": float(np.std(precision)),
                "recall_mean": float(np.mean(recall)),
                "recall_std": float(np.std(recall)),
                "f1_score_mean": float(np.mean(f1)),
                "f1_score_std": float(np.std(f1)),
                "support_mean": float(np.mean(support)),
                "support_std": float(np.std(support)),
            }
        )

    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError("Cannot infer CSV header from empty rows; provide fieldnames.")

    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def build_classification_metrics(rows: list[dict[str, Any]], labels: list[str] | None = None) -> dict[str, Any]:
    """Sukuria klasifikacines metrikas is top-1 retrieval prognoziu"""
    y_true = [str(row["true_label"]) for row in rows]
    y_pred = [str(row["top1_predicted_label"]) for row in rows]

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    else:
        labels = [str(label) for label in labels]

    if not y_true:
        empty_cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        return {
            "labels": labels,
            "classification_report_rows": [],
            "per_class_metrics_rows": [],
            "aggregate_metrics": {
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "weighted_precision": 0.0,
                "weighted_recall": 0.0,
                "weighted_f1": 0.0,
            },
            "confusion_matrix": empty_cm,
        }

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    classification_report_rows: list[dict[str, Any]] = []
    per_class_metrics_rows: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        classification_report_rows.append(
            {
                "label": label,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
        per_class_metrics_rows.append(
            {
                "label": label,
                "support": int(support[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
            }
        )

    classification_report_rows.extend(
        [
            {
                "label": "macro avg",
                "precision": float(report["macro avg"]["precision"]),
                "recall": float(report["macro avg"]["recall"]),
                "f1_score": float(report["macro avg"]["f1-score"]),
                "support": int(report["macro avg"]["support"]),
            },
            {
                "label": "weighted avg",
                "precision": float(report["weighted avg"]["precision"]),
                "recall": float(report["weighted avg"]["recall"]),
                "f1_score": float(report["weighted avg"]["f1-score"]),
                "support": int(report["weighted avg"]["support"]),
            },
        ]
    )

    return {
        "labels": labels,
        "classification_report_rows": classification_report_rows,
        "per_class_metrics_rows": per_class_metrics_rows,
        "aggregate_metrics": {
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
        },
        "confusion_matrix": cm,
    }


def export_embedding_tflite(embedding_model: tf.keras.Model, output_path: Path, fp16: bool = False) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(embedding_model)
    if fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)


def save_reference_json(
    output_path: Path,
    reference_db: dict[str, dict[str, Any]],
    embedding_dim: int,
    source_split: str,
    model_path: str,
    img_size: int,
    mode: str,
    max_refs_per_class: int,
) -> None:
    classes = []
    for label in sorted(reference_db):
        payload = reference_db[label]
        vectors = payload["vectors"]
        centroid = payload["centroid"]

        refs = [
            {"id": f"{label}__{i:04d}", "embedding": np.asarray(v, dtype=np.float32).tolist()}
            for i, v in enumerate(vectors, start=1)
        ]

        classes.append(
            {
                "label": label,
                "sample_count": len(vectors),
                "centroid_embedding": np.asarray(centroid, dtype=np.float32).tolist(),
                "references": refs,
                # Suderinamumas su senesniu centroid JSON
                "embedding": np.asarray(centroid, dtype=np.float32).tolist(),
            }
        )

    payload = {
        "version": 3,
        "method": "retrieval_embedding_experiment",
        "reference_mode": mode,
        "source_split": source_split,
        "model": model_path,
        "img_size": img_size,
        "embedding_size": embedding_dim,
        "max_refs_per_class": max_refs_per_class,
        "preprocessing": {
            "exif_transpose": True,
            "resize": [img_size, img_size],
            "channel_order": "RGB",
            "dtype": "float32",
            "input_scale": "0..255",
            "l2_normalize_embedding": True,
            "similarity_metric": "cosine",
        },
        "classes": classes,
    }

    save_json(payload, output_path)
