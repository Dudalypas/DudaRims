# Ratlankių stiliaus atpažinimo ir suderinamumo tikrinimo mobilioji programėlė

## Trumpas aprašymas

Ši Flutter programėlė pagal ratlankio nuotrauką nustato ratlankio stilių ir pateikia artimiausius kandidatus pagal požymių vektorių panašumą. Naudotojas gali pasirinkti automobilį ir patikrinti techninį ratlankio suderinamumą pagal pagrindinius parametrus. Pagrindinė atpažinimo ir suderinamumo tikrinimo logika vykdoma lokaliai įrenginyje.

## Realizacijos vaizdo įrašas

[Realizacijos vaizdo įrašas](https://youtube.com/shorts/jtNccEIVwjc?feature=share)

## Pagrindinis funkcionalumas

- ratlankio nuotraukos pateikimas iš kameros arba galerijos;
- ratlankio srities aptikimas;
- ratlankio vaizdo paruošimas;
- požymių vektoriaus generavimas;
- panašumo paieška pagal etaloninius vektorius;
- pagrindinio kandidato arba alternatyvų pateikimas;
- automobilio pasirinkimas;
- suderinamumo tikrinimas pagal PCD, CB, ET, skersmenį ir plotį;
- veikimas be interneto ryšio pagrindinėms funkcijoms.

## Naudotos technologijos

- Flutter / Dart – mobiliosios programėlės realizacija;
- TFLite – DI modelių vykdymas įrenginyje;
- YOLO tipo modelis – ratlankio srities aptikimas;
- MobileNetV3Small pagrindu paruoštas požymių vektoriaus modelis – vizualinių požymių išgavimas;
- SQLite – lokalūs ratlankių ir automobilių techniniai duomenys;
- Python – duomenų paruošimo, modelių eksportavimo ir vertinimo skriptai;
- Maestro – E2E ir stabilumo testavimo scenarijai.

## Projekto struktūra

- `lib/` – programėlės kodas;
- `assets/models/` – TFLite modeliai;
- `assets/data/` – SQLite DB, etaloniniai vektoriai ir kiti programėlės duomenys;
- `assets/data/rim_thumbnails/` – ratlankių iliustracijos;
- `test/` – automatiniai testai;
- `maestro/` – E2E ir stabilumo testavimo scenarijai;
- `xtools/` – duomenų, modelių ir vertinimo skriptai;
- `xdetection/` – ratlankio srities aptikimo modelio paruošimo skriptai;
- `docs/` – papildomi rezultatai ir vizualizacijos.

## Paleidimas

```powershell
flutter pub get
flutter run
```

Jei reikia paleisti konkrečiame Android įrenginyje:

```powershell
flutter run -d <device_id>
```

## Testavimas

```powershell
flutter analyze
flutter test
```

## Duomenų bazės ir modelių paruošimas

Galutinė programėlė naudoja lokalią SQLite duomenų bazę `assets/data/fitment.sqlite3`. Techniniai duomenys paruošiami pusiau automatizuotu procesu: dalis pradinių įrašų surenkama skriptais, o neaiškūs arba konfliktiški įrašai pažymimi rankinei peržiūrai.

Galutinė DB generuojama per `xtools/import_json_to_sqlite.py`. Etaloniniai požymių vektoriai generuojami per `xtools/generate_reference_embeddings.py`. Modeliai eksportuojami į TFLite ir naudojami programėlėje kaip ištekliai.

## Svarbiausi programėlės ištekliai

- `assets/models/0421best_float16.tflite`
- `assets/models/wheel_embedding_experiment_v2_float32.tflite`
- `assets/data/fitment.sqlite3`
- `assets/data/wheel_reference_embeddings_experiment_v2.json`
- `assets/data/rim_thumbnails/`

## Pastabos dėl projekto ribų

- Tai akademinis prototipas, ne komerciniam naudojimui paruoštas produktas.
- Duomenų aprėptis ribota ir priklauso nuo turimų šaltinių.
- Tolesniam vystymui numatomas duomenų ir modelių atnaujinimo mechanizmas, iOS versijos testavimas ir galimas papildytos realybės funkcionalumas.
