import '../core/constants/app_constants.dart';
import 'recognition_candidate.dart';

enum RecognitionBranch { strong, candidates, failure }

enum RecognitionManualPickReason { lowTop1Similarity, smallTop12Margin }

class RecognitionOutcome {
  final List<RecognitionCandidate> top5;
  final String? failureReason;
  final RecognitionManualPickReason? manualPickReason;

  const RecognitionOutcome({
    required this.top5,
    this.failureReason,
    this.manualPickReason,
  });

  factory RecognitionOutcome.failure({
    String? reason,
    List<RecognitionCandidate> top5 = const [],
  }) {
    return RecognitionOutcome(
      top5: top5,
      failureReason: reason,
      manualPickReason: null,
    );
  }

  factory RecognitionOutcome.manualPick({
    required List<RecognitionCandidate> topCandidates,
    required RecognitionManualPickReason reason,
  }) {
    return RecognitionOutcome(
      top5: topCandidates,
      failureReason: null,
      manualPickReason: reason,
    );
  }

  bool get hasTopPrediction => top5.isNotEmpty;
  RecognitionCandidate? get top1OrNull => hasTopPrediction ? top5.first : null;
  RecognitionCandidate get top1 => top5.first;
  List<RecognitionCandidate> get top3 => top5.take(3).toList(growable: false);

  RecognitionBranch get branch {
    if (manualPickReason != null && top5.isNotEmpty) {
      return RecognitionBranch.candidates;
    }

    if ((failureReason ?? '').trim().isNotEmpty) {
      return RecognitionBranch.failure;
    }

    final candidate = top1OrNull;
    if (candidate == null) {
      return RecognitionBranch.failure;
    }

    if (candidate.score < AppConstants.recognitionLowSimilarityThreshold) {
      return RecognitionBranch.failure;
    }
    if (candidate.score >= AppConstants.recognitionStrongSimilarityThreshold) {
      return RecognitionBranch.strong;
    }
    return RecognitionBranch.candidates;
  }

  bool get isConfident => branch == RecognitionBranch.strong;

  String get candidateHintMessage {
    switch (manualPickReason) {
      case RecognitionManualPickReason.smallTop12Margin:
        return AppConstants.recognitionAmbiguousCandidatesMessage;
      case RecognitionManualPickReason.lowTop1Similarity:
      case null:
        return 'Panašumo įvertis žemesnis, pasirinkite vieną iš galimų variantų.';
    }
  }
}
