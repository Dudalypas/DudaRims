enum RecognitionPipelineMode { embeddingRetrieval, classifierBaseline }

enum RetrievalReferenceMode { multiReference, centroidBaseline }

enum ClassAggregationMode { maxSimilarity, topNSimilarityAverage }

class AppConstants {
  // Similarity thresholds for retrieval-style recognition branch selection.
  static const double recognitionStrongSimilarityThreshold = 0.82;
  static const double recognitionLowSimilarityThreshold = 0.15;

  static const String recognitionFailureMessage =
      'Pabandyk įkelti aiškesnę nuotrauką, kurioje ratlankis matomas iš priekio ir užima didžiąją kadro dalį.';

  // Greitas jungiklis UI testams, kai false, galima testuoti flow be realaus ML
  static const bool enableRealMlInference = true;

  // Migration strategy: use embedding retrieval by default, keep classifier path as baseline.
  static const RecognitionPipelineMode recognitionPipelineMode =
      RecognitionPipelineMode.embeddingRetrieval;

  // Fallback to classifier when embedding retrieval path fails or has no candidates.
  static const bool enableClassifierFallback = true;
  static const bool enablePipelineDebugLogs = true;

  static const String embeddingModelAsset =
      'assets/models/wheel_embedding_cropped_float32.tflite';
  static const String referenceEmbeddingsAsset =
      'assets/data/wheel_reference_embeddings.json';

  // Retrieval behavior configuration.
  static const RetrievalReferenceMode retrievalReferenceMode =
      RetrievalReferenceMode.multiReference;
  static const ClassAggregationMode classAggregationMode =
      ClassAggregationMode.topNSimilarityAverage;
  static const int classAggregationTopN = 3;

  // Post-detection crop refinement knobs for retrieval pipeline.
  static const double retrievalCropPaddingRatio = 0.04;
  static const double retrievalCropTightenRatio = 0.94;
  static const bool retrievalCropEnforceSquare = true;

  // FitmentTolerancijos, virsijus rodome perspejima, bet ne kritini
  static const double fitmentDiameterWarningDeviationIn = 1.0;
  static const double fitmentWidthWarningDeviationJ = 0.5;
  static const double fitmentEtWarningDeviationMm = 5.0;
}
