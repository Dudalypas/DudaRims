enum RecognitionPipelineMode { embeddingRetrieval, classifierBaseline }

enum RetrievalReferenceMode { multiReference, centroidBaseline }

enum ClassAggregationMode { maxSimilarity, topNSimilarityAverage }

class AppConstants {
  // Similarity thresholds for retrieval-style recognition branch selection.
  static const double recognitionStrongSimilarityThreshold = 0.82;
  static const double recognitionLowSimilarityThreshold = 0.15;

  static const String recognitionFailureMessage =
      'Pabandyk įkelti aiškesnę nuotrauką, kurioje ratlankis matomas iš priekio ir užima didžiąją kadro dalį.';
  static const String recognitionNoRimDetectedMessage =
      'Nepavyko aptikti ratlankio nuotraukoje.';
  static const String recognitionUncertainResultMessage =
      'Nepavyko patikimai atpažinti ratlankio.';

  // Greitas jungiklis UI testams, kai false, galima testuoti flow be realaus ML
  static const bool enableRealMlInference = true;

  // Migration strategy: use embedding retrieval by default, keep classifier path as baseline.
  static const RecognitionPipelineMode recognitionPipelineMode =
      RecognitionPipelineMode.embeddingRetrieval;

  // Fallback to classifier when embedding retrieval path fails or has no candidates.
  static const bool enableClassifierFallback = true;
  static const bool enablePipelineDebugLogs = true;
    static const bool enableOriginalImageEmbeddingFallbackForDebug = false;

  static const String embeddingModelAsset =
      'assets/models/wheel_embedding_experiment_v2_float32.tflite';
  static const String referenceEmbeddingsAsset =
      'assets/data/wheel_reference_embeddings_experiment_v2.json';

  // Retrieval behavior configuration.
  static const RetrievalReferenceMode retrievalReferenceMode =
      RetrievalReferenceMode.centroidBaseline;
  static const ClassAggregationMode classAggregationMode =
      ClassAggregationMode.topNSimilarityAverage;
  static const int classAggregationTopN = 1;

 // Detector rejection thresholds.
static const double detectorMinScoreThreshold = 0.60;       // buvo 0.65
static const double detectorMinBboxAreaRatio = 0.010;       // buvo 0.015
static const double detectorMaxBboxAreaRatio = 0.80;        // buvo 0.70
static const double? detectorMaxAspectRatio = 3.0;          // buvo 2.5
static const double detectorHardMinScoreThreshold = 0.35;   // buvo 0.45
static const double detectorHardMinBboxAreaRatio = 0.003;   // buvo 0.005
static const double detectorHardMaxAspectRatio = 5.0;       // buvo 4.0

// Controlled centered fallback for rim-only images.
static const bool enableCenteredRimOnlyFallback = true;
static const double rimOnlyFallbackCenteredCropRatio = 0.86;      // buvo 0.80
static const double rimOnlyFallbackCenterEnergyMinRatio = 0.52;   // buvo 0.62
static const double rimOnlyFallbackMinTop1Similarity = 0.78;      // buvo 0.86
static const double rimOnlyFallbackMinTop1Top2Margin = 0.015;     // buvo 0.05

// Retrieval rejection thresholds.
static const double retrievalMinTop1Similarity = 0.58;            // buvo 0.60
static const double retrievalMinTop1Top2Margin = 0.008;           // buvo 0.015

// Post-detection crop refinement knobs for retrieval pipeline.
static const double retrievalCropPaddingRatio = 0.03;             // buvo 0.04
static const double retrievalCropTightenRatio = 0.96;             // buvo 0.94
static const bool retrievalCropEnforceSquare = true;

  // FitmentTolerancijos, virsijus rodome perspejima, bet ne kritini
  static const double fitmentDiameterWarningDeviationIn = 1.0;
  static const double fitmentWidthWarningDeviationJ = 0.5;
  static const double fitmentEtWarningDeviationMm = 5.0;
}
