# Ratlankių atpažinimo ir suderinamumo pipeline

Šitas repozitorijos gabalas yra mano bakalauro darbo dalis. Idėja paprasta: iš skirtingų šaltinių susirinkti duomenis apie Skoda ratlankius, juos sutvarkyti į vieną normalų formatą, ir po to automatiškai patikrinti suderinamumą su automobilio modeliu.

Projektas turi 2 pagrindines kryptis:

- duomenų surinkimas ir sujungimas (Python skriptai)
- ratlankių atpažinimas pagal embedding + retrieval (ML dalis)

## Ką sugeneruoja pipeline

Pagrindiniai failai:

- `wheels.csv` ir `wheels.json` (ratlankiai)
- `skoda_models.csv` ir `skoda_models.json` (modelių fitment ribos)

Papildomi kontroliniai failai:

- `wheel_class_mapping.csv`
- `manual_review.csv`
- `source_conflicts.csv`
- `fitment_examples.json`

## Šaltinių prioritetas

1. Pirminis šaltinis: oficialūs Skoda aksesuarų/katalogo puslapiai.
2. Antrinis šaltinis: trečiųjų šalių fitment puslapiai (modelių riboms).

Svarbios taisyklės:

- trečiosios šalies reikšmės nelaikomos automatiškai „teisingesnėmis“ už oficialias;
- jei trūksta reikšmių arba jos konfliktuoja, įrašas žymimas rankinei peržiūrai;
- niekas nėra „išgalvojama“: geriau palikti tuščią ir pažymėti peržiūrai.

## Reikalavimai

- Python 3.11+
- paketai: `requests`, `beautifulsoup4`, `pandas`

Įsidiegti priklausomybes:

```powershell
pip install -r requirements.txt
```

## Naudojami pagrindiniai skriptai

- `scrape_skoda_wheels.py`
- `scrape_skoda_models.py`
- `scrape_tirewheelguide_models.py`
- `merge_fitment_data.py`
- `fitment_checker.py`

## 1) Oficialių ratlankių duomenų surinkimas

Paleidimas su sąrašo URL ir/arba konkrečių produktų URL:

```powershell
python scrape_skoda_wheels.py \
	--listing-url "https://example.skoda-official.com/accessories/wheels" \
	--product-url "https://example.skoda-official.com/accessories/wheel-braga" \
	--output-dir "." \
	--save-raw-html
```

Išėjimas:

- `wheels.csv`
- `wheels.json`
- `wheel_class_mapping.csv`

Pastabos:

- tokie matmenys kaip `7.0J x 17 ET 49` išskaidomi į `width_j`, `diameter_in`, `et`;
- `pcd`, `cb`, `bolt_count` paliekami tušti, jei aiškiai nerasta;
- kai trūksta kritinių laukų, nustatomas `needs_manual_review=True`.

## 2) Skoda modelių fitment duomenys iš TireWheelGuide

Šitas skriptas eina per generacijos bloką, surenka visas modifikacijų lenteles ir padaro vieną galutinę eilutę vienai generacijai.

Pavyzdinis šaltinis:

- `https://tirewheelguide.com/sizes/skoda/octavia/1996/`

Skaičiuojamos ribos:

- `diameter_min_in`, `diameter_max_in`
- `width_min_j`, `width_max_j`
- `et_min`, `et_max`

Taip pat normalizuojami:

- `pcd`, `cb`, `thread_size`, `bolt_count`, `center_bore_mm`

Kai duodami keli URL, duomenys sujungiami pagal raktą:

- `brand`
- `model`
- `generation`
- `year_from`
- `year_to`

Taip išvengiama dublikatų, kai skirtingi metų puslapiai rodo tą pačią generaciją.

### Vieno URL režimas

```powershell
python scrape_tirewheelguide_models.py \
	--source-url "https://tirewheelguide.com/sizes/skoda/octavia/1996/" \
	--output-dir ".\fitment_data" \
	--verbose
```

### Kelių URL režimas

```powershell
python scrape_tirewheelguide_models.py \
	--source-url "https://tirewheelguide.com/sizes/skoda/octavia/1996/" \
	--source-url "https://tirewheelguide.com/sizes/skoda/octavia/2004/" \
	--output-dir ".\fitment_data" \
	--verbose
```

### Raw HTML išsaugojimas

```powershell
python scrape_tirewheelguide_models.py \
	--source-url "https://tirewheelguide.com/sizes/skoda/octavia/1996/" \
	--save-raw-html \
	--raw-html-dir ".\raw_html\tirewheelguide" \
	--output-dir ".\fitment_data" \
	--verbose
```

### CSV įvesties režimas (`--pairs-csv` arba `--input-csv`)

CSV stulpeliai:

- `brand`
- `model`
- `source_url`

Pavyzdys:

```csv
brand,model,source_url
Skoda,Octavia,https://tirewheelguide.com/sizes/skoda/octavia/1996/
Skoda,Octavia,https://tirewheelguide.com/sizes/skoda/octavia/2004/
```

Paleidimas:

```powershell
python scrape_tirewheelguide_models.py \
	--pairs-csv "model_source_pairs.csv" \
	--output-dir ".\fitment_data" \
	--verbose
```

Išėjimas:

- `skoda_models.csv`
- `skoda_models.json`
- `source_conflicts.csv`
- `manual_review.csv`

Pastabos:

- šiame kelyje nenaudojamas `Playwright`;
- `Wheel-Size/wheelfitment` scraping čia nereikalingas;
- trūkstami kritiniai laukai žymimi rankinei peržiūrai, o ne „atspėjami“;
- konfliktai vienoje generacijoje sprendžiami pagal dažniausiai pasikartojančią reikšmę ir pažymimi `notes`;
- pasikartojančios generacijos iš kelių URL sujungiamos į vieną eilutę, `notes` lauke pridedant `merged_from_urls=<n>`.

## 3) Sujungimas, normalizavimas ir validacija

```powershell
python merge_fitment_data.py \
	--wheels "wheels.csv" \
	--models ".\fitment_data\skoda_models.csv" \
	--mapping "wheel_class_mapping.csv" \
	--output-dir ".\fitment_data"
```

Atnaujinami failai:

- `wheels.csv` / `wheels.json`
- `skoda_models.csv` / `skoda_models.json`
- `manual_review.csv`
- `source_conflicts.csv`

Į `manual_review.csv` patenka atvejai, kai yra:

- konfliktuojančios ET reikšmės;
- konfliktuojančios modelio generacijos;
- trūksta PCD/CB/thread/width/diameter;
- dviprasmiškas ratlankio klasės priskyrimas.

## 4) Suderinamumo tikrinimas

`fitment_checker.py` turi funkciją:

```python
check_fitment(wheel: dict, vehicle: dict) -> dict
```

Grąžinama struktūra:

- `result`: `fits`, `caution`, `does_not_fit`
- `reasons`: paaiškinimų sąrašas

Sprendimo logika:

- PCD nesutampa -> `does_not_fit`
- ratlankio CB mažesnis nei automobilio reikalaujamas -> `does_not_fit`
- ratlankio CB didesnis -> `caution` (gali reikėti centravimo žiedų)
- diametras už leidžiamų ribų -> `does_not_fit`
- plotis už ribų -> nedidelis nukrypimas `caution`, didesnis `does_not_fit`
- ET už ribų -> nedidelis nukrypimas `caution`, didesnis `does_not_fit`
- trūksta kritinių duomenų -> `caution` su aiškiu paaiškinimu

Greitas pavyzdžių generavimas:

```powershell
python fitment_checker.py --wheels "wheels.csv" --models "skoda_models.csv" --output "fitment_examples.json" --limit 5
```

## Duomenų kilmė ir rankinė peržiūra

Kiekvienas įrašas turi kilmės laukus:

- `source_type`
- `source_page`
- `extraction_confidence`
- `needs_manual_review`

Tai leidžia saugiai naudoti duomenis Flutter programėlėje arba lokalioje JSON/SQLite saugykloje.

## Embedding + retrieval dalis (bakalaurui)

Atpažinimo eiga projekte:

1. ratlankio aptikimas
2. crop patikslinimas po aptikimo
3. požymių vektoriaus (embedding) išgavimas
4. cosine similarity paieška tarp etaloninių vektorių
5. top-k kandidatų grąžinimas
6. fitment logika lieka ta pati

### 1) Embedding modelio eksportas iš esamo Keras klasifikatoriaus

Naudojamas esamas `best.keras` backbone, paimamas priešpaskutinis sluoksnis su L2 normalizacija.

```powershell
python xtools\export_embedding_model.py \
	--source-model "assets\models\best.keras" \
	--output "assets\models\wheel_embedding_cropped_float32.tflite"
```

Pasirinktinis fp16 eksportas:

```powershell
python xtools\export_embedding_model.py \
	--source-model "assets\models\best.keras" \
	--output "assets\models\wheel_embedding_cropped_float32.tflite" \
	--export-fp16
```

### 2) Etaloninių vektorių kūrimas (tik TRAIN daliai)

2 fazėje numatytas multi-reference retrieval, o centroid paliktas kaip legacy baseline.

Multi-reference JSON generavimas:

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

Kiti reference režimai:

- `--reference-mode all` (visi train pavyzdžiai kiekvienai klasei)
- `--reference-mode centroid` (legacy centroid bazė)

Pasirinktinis detector crop per etalonų generavimą:

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

### 3) Retrieval kokybės vertinimas (Recall@1/3/5)

Vienu paleidimu palyginami centroid ir multi-reference režimai, su pasirinktiniais crop profiliais.

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

Greitas smoke test režimas:

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

Rezultatai:

- `trained_cropped_classifier/retrieval_eval_summary.json`
- `trained_cropped_classifier/retrieval_eval_per_class.csv`
- `trained_cropped_classifier/retrieval_eval_experiments.csv`
- `trained_cropped_classifier/retrieval_eval_top1_compare.csv`

### 4) Crop pavyzdžių išsaugojimas vizualiai peržiūrai

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

### Preprocessing nuoseklumas (mokymas / eksportas / inferencija)

Dabartinės bendros prielaidos visame pipeline:

- pritaikoma EXIF orientacija;
- naudojama RGB kanalų tvarka;
- resize į `224x224`;
- įvesties tipas `float32`;
- pikselių skalė `0..255` (be papildomo MobileNetV3 preprocessing sluoksnio);
- embedding išėjimas yra L2-normalizuotas;
- retrieval metrika: cosine similarity.

## Prielaidos

- oficialių puslapių struktūra gali skirtis pagal regioną ir kalbą, todėl parsing yra atsargus;
- trečiųjų šalių fitment puslapiai naudojami modelių riboms užpildyti;
- jei reikšmė neištraukiama patikimai, ji paliekama tuščia ir žymima peržiūrai.

## Ribotumai

- dalis puslapių kraunasi dinamiškai; pagal nutylėjimą čia vengiama Selenium;
- pasikeitus HTML struktūrai gali tekti taisyti parserio selektorius;
- `wheel_class_mapping` vietomis gali likti dviprasmiškas (ypač generic klasėms kaip `Steel_Wheel`).
