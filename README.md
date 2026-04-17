# Skoda Wheel Ingestion And Fitment Pipeline

This repository now includes a small Python ingestion project for bachelor thesis work on Skoda wheel recognition and compatibility checking.

The ingestion pipeline builds two normalized datasets:

- `wheels.csv` + `wheels.json` from **official Skoda accessory pages** (primary)
- `skoda_models.csv` + `skoda_models.json` from **TireWheelGuide generation-level parsing** (secondary)

It also generates:

- `wheel_class_mapping.csv`
- `manual_review.csv`
- `source_conflicts.csv`
- `fitment_examples.json`

## Source Hierarchy

1. **Primary:** official Skoda accessory/catalog pages for wheel specs.
2. **Secondary:** third-party fitment references for vehicle model fitment ranges.

Rules used by scripts:

- Third-party values are never silently treated as more authoritative than official values.
- Missing or conflicting values are flagged for manual review.
- No fabricated technical values: blank + review flag is preferred over guessing.

## Python Requirements

- Python 3.11+
- Packages: `requests`, `beautifulsoup4`, `pandas`

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Files Added

- `scrape_skoda_wheels.py`
- `scrape_skoda_models.py`
- `scrape_tirewheelguide_models.py`
- `merge_fitment_data.py`
- `fitment_checker.py`

## 1) Scrape Official Wheel Data

Run with official listing and/or direct product URLs.

```powershell
python scrape_skoda_wheels.py \
	--listing-url "https://example.skoda-official.com/accessories/wheels" \
	--product-url "https://example.skoda-official.com/accessories/wheel-braga" \
	--output-dir "." \
	--save-raw-html
```

Output:

- `wheels.csv`
- `wheels.json`
- `wheel_class_mapping.csv`

Notes:

- Rim dimensions like `7.0J x 17 ET 49` are parsed into `width_j`, `diameter_in`, `et`.
- `pcd`, `cb`, and `bolt_count` are left blank when not explicitly present.
- `needs_manual_review=True` is set when critical fields are missing.

## 2) Scrape Skoda Model Fitment Specs (TireWheelGuide-first)

Vehicle ingestion now has a dedicated TireWheelGuide scraper that aggregates all modification tables under each generation block and emits one final row per generation.

Source page example:

- `https://tirewheelguide.com/sizes/skoda/octavia/1996/`

The scraper inspects all modification sections and all fitment tables in the generation block to compute:

- `diameter_min_in`, `diameter_max_in`
- `width_min_j`, `width_max_j`
- `et_min`, `et_max`

It also extracts and normalizes:

- `pcd`, `cb`, `thread_size`, `bolt_count`, `center_bore_mm`

After scraping multiple URLs, rows are consolidated by generation key:

- `brand`
- `model`
- `generation`
- `year_from`
- `year_to`

This prevents duplicate generation rows when several year pages resolve to the same generation.

### Single URL mode

```powershell
python scrape_tirewheelguide_models.py \
	--source-url "https://tirewheelguide.com/sizes/skoda/octavia/1996/" \
	--output-dir ".\fitment_data" \
	--verbose
```

### Multiple URL mode

```powershell
python scrape_tirewheelguide_models.py \
	--source-url "https://tirewheelguide.com/sizes/skoda/octavia/1996/" \
	--source-url "https://tirewheelguide.com/sizes/skoda/octavia/2004/" \
	--output-dir ".\fitment_data" \
	--verbose

### Save raw HTML snapshots

```powershell
python scrape_tirewheelguide_models.py \
	--source-url "https://tirewheelguide.com/sizes/skoda/octavia/1996/" \
	--save-raw-html \
	--raw-html-dir ".\raw_html\tirewheelguide" \
	--output-dir ".\fitment_data" \
	--verbose
```
```

### CSV input mode (`--pairs-csv` or `--input-csv`)

CSV columns:

- `brand`
- `model`
- `source_url`

Example:

```csv
brand,model,source_url
Skoda,Octavia,https://tirewheelguide.com/sizes/skoda/octavia/1996/
Skoda,Octavia,https://tirewheelguide.com/sizes/skoda/octavia/2004/
```

Run:

```powershell
python scrape_tirewheelguide_models.py \
	--pairs-csv "model_source_pairs.csv" \
	--output-dir ".\fitment_data" \
	--verbose
```

Output:

- `skoda_models.csv`
- `skoda_models.json`
- `source_conflicts.csv`
- `manual_review.csv`

Notes:

- No Playwright is used in this ingestion path.
- No Wheel-Size/wheelfitment scraping is required for this workflow.
- Missing critical fields are flagged for manual review instead of guessed.
- Conflicting values inside one generation are resolved by most-common value and flagged in `notes`.
- Duplicate generation rows across multiple source URLs are merged into one row with joined `source_page` and `merged_from_urls=<n>` in `notes`.

## 3) Merge, Normalize, Validate, And Build Review Queue

```powershell
python merge_fitment_data.py \
	--wheels "wheels.csv" \
	--models ".\fitment_data\skoda_models.csv" \
	--mapping "wheel_class_mapping.csv" \
	--output-dir ".\fitment_data"
```

Outputs refreshed and validated:

- `wheels.csv` / `wheels.json`
- `skoda_models.csv` / `skoda_models.json`
- `manual_review.csv`
- `source_conflicts.csv`

Manual review rows are generated for:

- conflicting ET values
- conflicting model generations
- missing PCD/CB/thread/width/diameter fields
- ambiguous wheel-class mappings

## 4) Compatibility Checker

`fitment_checker.py` exposes:

```python
check_fitment(wheel: dict, vehicle: dict) -> dict
```

Returned shape:

- `result`: `fits`, `caution`, `does_not_fit`
- `reasons`: list of explanations

Decision rules:

- PCD mismatch => `does_not_fit`
- wheel CB smaller than vehicle requirement => `does_not_fit`
- wheel CB larger than vehicle => `caution` (hub-centric rings)
- diameter outside allowed range => `does_not_fit`
- width outside range => slight = `caution`, clear = `does_not_fit`
- ET outside range => slight = `caution`, clear = `does_not_fit`
- missing critical data => `caution` with explicit explanation

Create quick example outputs:

```powershell
python fitment_checker.py --wheels "wheels.csv" --models "skoda_models.csv" --output "fitment_examples.json" --limit 5
```

## Provenance And Manual Review

Every record keeps source attribution:

- `source_type`
- `source_page`
- `extraction_confidence`
- `needs_manual_review`

This is designed so records can be safely consumed by a Flutter app or local JSON/SQLite store later.

## Embedding Retrieval Pipeline (Thesis Alignment)

This project now supports a retrieval-style recognition path:

1. wheel detection
2. post-detection crop refinement
3. feature vector extraction (embedding)
4. cosine-similarity search against local reference vectors
5. top-k retrieval for candidate wheel classes
6. fitment/compatibility logic remains unchanged

### 1) Export embedding model from existing Keras classifier

Uses the existing `best.keras` backbone and exposes the penultimate representation with L2 normalization.

```powershell
python xtools\export_embedding_model.py \
	--source-model "assets\models\best.keras" \
	--output "assets\models\wheel_embedding_cropped_float32.tflite"
```

Optional fp16 export:

```powershell
python xtools\export_embedding_model.py \
	--source-model "assets\models\best.keras" \
	--output "assets\models\wheel_embedding_cropped_float32.tflite" \
	--export-fp16
```

### 2) Build reference vectors (TRAIN split only)

Phase 2 uses multi-reference retrieval by default and keeps centroid as legacy baseline.

Generate multi-reference JSON (limited references per class + optional centroid for baseline comparison):

```powershell
python xtools\generate_reference_embeddings.py \
	--dataset-root "C:\Users\vilja\Desktop\Training_Mixed_V1" \
	--split train \
	--model "assets\models\wheel_embedding_cropped_float32.tflite" \
	--output "assets\data\wheel_reference_embeddings.json" \
	--reference-mode limited \
	--max-refs-per-class 20 \
	--sampling random \
	--seed 42 \
	--include-centroid
```

Other supported reference modes:

- `--reference-mode all` (all train references per class)
- `--reference-mode centroid` (legacy centroid-per-class baseline)

Optional detector-based crop refinement during reference generation:

```powershell
python xtools\generate_reference_embeddings.py \
	--dataset-root "C:\Users\vilja\Desktop\Training_Mixed_V1" \
	--split train \
	--model "assets\models\wheel_embedding_cropped_float32.tflite" \
	--output "assets\data\wheel_reference_embeddings.json" \
	--reference-mode limited \
	--max-refs-per-class 20 \
	--use-detector-crop \
	--detector-model "assets\models\best_float16.tflite" \
	--crop-padding-ratio 0.04 \
	--crop-tighten-ratio 0.94 \
	--crop-enforce-square \
	--save-crops-dir "trained_cropped_classifier\crop_debug_refs"
```

### 3) Evaluate retrieval quality (Recall@1/3/5) with mode comparison

Evaluates centroid baseline and multi-reference modes in one run, with optional crop-profile sweeps.

```powershell
python xtools\evaluate_retrieval.py \
	--dataset-root "C:\Users\vilja\Desktop\Training_Mixed_V1" \
	--model "assets\models\wheel_embedding_cropped_float32.tflite" \
	--reference-json "assets\data\wheel_reference_embeddings.json" \
	--splits val test \
	--retrieval-modes centroid multi_max multi_topn_avg \
	--topn-values 3 5 \
	--crop-profiles "no_crop:false:0.04:0.94:true" "det_crop:true:0.04:0.94:true" \
	--detector-model "assets\models\best_float16.tflite"
```

Quick smoke-check mode (fast):

```powershell
python xtools\evaluate_retrieval.py \
	--dataset-root "C:\Users\vilja\Desktop\Training_Mixed_V1" \
	--model "assets\models\wheel_embedding_cropped_float32.tflite" \
	--reference-json "assets\data\wheel_reference_embeddings.json" \
	--splits val \
	--retrieval-modes centroid multi_max multi_topn_avg \
	--topn-values 3 \
	--crop-profiles "no_crop:false:0.04:0.94:true" \
	--limit-per-class 1
```

Outputs:

- `trained_cropped_classifier/retrieval_eval_summary.json`
- `trained_cropped_classifier/retrieval_eval_per_class.csv`
- `trained_cropped_classifier/retrieval_eval_experiments.csv`
- `trained_cropped_classifier/retrieval_eval_top1_compare.csv`

### 4) Save crop refinement examples for visual inspection

```powershell
python xtools\debug_refined_crops.py \
	--input-root "C:\Users\vilja\Desktop\Training_Mixed_V1\val" \
	--output-dir "trained_cropped_classifier\crop_debug" \
	--detector-model "assets\models\best_float16.tflite" \
	--per-class 8 \
	--crop-padding-ratio 0.04 \
	--crop-tighten-ratio 0.94 \
	--crop-enforce-square
```

### Preprocessing Consistency (Training/Export/Inference)

Current unified assumptions across export, reference generation, evaluation, and Flutter inference:

- EXIF orientation is applied
- RGB channel order is used
- resize to `224x224`
- `float32` input dtype
- input pixel scale is `0..255` (no additional MobileNetV3 preprocessing layer)
- output embedding is L2-normalized
- retrieval similarity metric is cosine similarity

## Assumptions

- Official pages may vary by region and language; parsing is heuristic and conservative.
- Third-party fitment sites are used to bootstrap model specs only.
- When parsers cannot confidently extract a value, they leave it blank and flag review.

## Limitations

- Some websites load data dynamically; this implementation avoids Selenium by default.
- HTML structures can change and require parser selector updates.
- Wheel-class mapping confidence can remain ambiguous for style variants and generic classes (for example `Steel_Wheel`)
