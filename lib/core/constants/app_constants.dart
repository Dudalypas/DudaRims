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
  static const String recognitionAmbiguousCandidatesMessage =
      'Rasti keli labai panašūs variantai. Pasirinkite tiksliausią.';

  // Greitas jungiklis UI testams, kai false, galima testuoti flow be realaus ML
  static const bool enableRealMlInference = true;

  static const RecognitionPipelineMode recognitionPipelineMode =
      RecognitionPipelineMode.embeddingRetrieval;

  // Fallback classifieri
  static const bool enableClassifierFallback = false;
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
    static const double detectorMinScoreThreshold = 0.60;
    static const double detectorMinBboxAreaRatio = 0.0035;
    static const double detectorMaxBboxAreaRatio = 1.00;
    static const double detectorMaxAspectRatio = 3.0;
    static const double detectorHardMinScoreThreshold = 0.35;
    static const double detectorHardMinBboxAreaRatio = 0.002;
    static const double detectorHardMaxAspectRatio = 5.0;

    // Controlled centered fallback for rim-only images.
    static const bool enableCenteredRimOnlyFallback = false;
    static const double rimOnlyFallbackCenteredCropRatio = 0.86;
    static const double rimOnlyFallbackCenterEnergyMinRatio = 0.52;
    static const double rimOnlyFallbackMinTop1Similarity = 0.78;
    static const double rimOnlyFallbackMinTop1Top2Margin = 0.015;

    // Retrieval rejection thresholds.
    static const double retrievalMinTop1Similarity = 0.56;
    static const double retrievalMinTop1Top2Margin = 0.008;

    // Post-detection crop refinement knobs for retrieval pipeline.
    static const double retrievalCropPaddingRatio = 0.08;
    static const double retrievalCropTightenRatio = 1.00;
    static const bool retrievalCropEnforceSquare = true;

    // Optional post-crop mask before embedding inference.
    static const bool enableRetrievalEllipseMask = true;
    static const double retrievalEllipseMaskInsetRatio = 0.07;
    static const double retrievalEllipseMaskFeather = 0.05;
    static const bool retrievalEllipseMaskUseCircleFallback = false;

  // FitmentTolerancijos, virsijus rodome perspejima, bet ne kritini
  static const double fitmentDiameterWarningDeviationIn = 1.0;
  static const double fitmentWidthWarningDeviationJ = 0.5;
  static const double fitmentEtWarningDeviationMm = 5.0;
}
