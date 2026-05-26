enum RetrievalReferenceMode { multiReference, centroidBaseline }

enum ClassAggregationMode { maxSimilarity, topNSimilarityAverage }

class AppConstants {
  // Naudotojui rodomi atpazinimo pranesimai
  static const String recognitionFailureMessage =
      'Pabandyk įkelti aiškesnę nuotrauką, kurioje ratlankis matomas iš priekio ir užima didžiąją kadro dalį.';
  static const String recognitionNoRimDetectedMessage =
      'Nepavyko aptikti ratlankio nuotraukoje.';
  static const String recognitionUncertainResultMessage =
      'Nepavyko patikimai atpažinti ratlankio.';
  static const String recognitionAmbiguousCandidatesMessage =
      'Rasti keli labai panašūs variantai. Pasirinkite tiksliausią.';

  // Jungiklis leidzia testuoti UI eiga be realaus ML vykdymo
  static const bool enableRealMlInference = true;

  static const bool enablePipelineDebugLogs = false;

  static const String embeddingModelAsset =
      'assets/models/wheel_embedding_experiment_v2_float32.tflite';
  static const String referenceEmbeddingsAsset =
      'assets/data/wheel_reference_embeddings_experiment_v2.json';

  // Etaloniniu pozymiu vektoriu naudojimo strategija
  static const RetrievalReferenceMode retrievalReferenceMode =
      RetrievalReferenceMode.centroidBaseline;
  static const ClassAggregationMode classAggregationMode =
      ClassAggregationMode.topNSimilarityAverage;
  static const int classAggregationTopN = 1;

  // Detektoriaus filtrai pries perduodant vaizda pozymiu modeliui
  static const double detectorMinScoreThreshold = 0.60;
  static const double detectorMinBboxAreaRatio = 0.0035;
  static const double detectorMaxBboxAreaRatio = 1.00;
  static const double detectorMaxAspectRatio = 3.0;

  // Grieztos ribos aiskiai netinkamiems aptikimams atmesti
  static const double detectorHardMinScoreThreshold = 0.35;
  static const double detectorHardMinBboxAreaRatio = 0.002;
  static const double detectorHardMaxAspectRatio = 5.0;

  // Panasumo slenksciai automatiniam rezultatui arba alternatyvoms parinkti
  static const double retrievalMinTop1Similarity = 0.56;
  static const double retrievalMinTop1Top2Margin = 0.008;
  static const double recognitionStrongSimilarityThreshold = 0.82;
static const double recognitionLowSimilarityThreshold = 0.15;

  // Eksperimentinis centrinio apkirpimo rezimas, galutineje grandineje isjungtas
  static const bool enableCenteredRimOnlyFallback = false;
  static const double rimOnlyFallbackCenteredCropRatio = 0.86;
  static const double rimOnlyFallbackCenterEnergyMinRatio = 0.52;
  static const double rimOnlyFallbackMinTop1Similarity = 0.78;
  static const double rimOnlyFallbackMinTop1Top2Margin = 0.015;

  // Apkirpimo parametrai po ratlankio srities aptikimo
  static const double retrievalCropPaddingRatio = 0.08;
  static const double retrievalCropTightenRatio = 1.00;
  static const bool retrievalCropEnforceSquare = true;

  // Elipses kauke 
  static const bool enableRetrievalEllipseMask = true;
  static const double retrievalEllipseMaskInsetRatio = 0.07;
  static const double retrievalEllipseMaskFeather = 0.05;
  static const bool retrievalEllipseMaskUseCircleFallback = false;

  // Suderinamumo nuokrypiai, kai rodoma pastaba, bet ne kritinis neatitikimas
  static const double fitmentDiameterWarningDeviationIn = 1.0;
  static const double fitmentWidthWarningDeviationJ = 0.5;
  static const double fitmentEtWarningDeviationMm = 5.0;
}