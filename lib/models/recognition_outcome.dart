import '../core/constants/app_constants.dart';
import 'recognition_candidate.dart';

enum RecognitionBranch { strong, candidates, failure }

class RecognitionOutcome {
  final List<RecognitionCandidate> top5;
  final String? failureReason;

  const RecognitionOutcome({required this.top5, this.failureReason});

  factory RecognitionOutcome.failure({
    String? reason,
    List<RecognitionCandidate> top5 = const [],
  }) {
    return RecognitionOutcome(top5: top5, failureReason: reason);
  }

  bool get hasTopPrediction => top5.isNotEmpty;
  RecognitionCandidate? get top1OrNull => hasTopPrediction ? top5.first : null;
  RecognitionCandidate get top1 => top5.first;
  List<RecognitionCandidate> get top3 => top5.take(3).toList(growable: false);

  RecognitionBranch get branch {
    if ((failureReason ?? '').trim().isNotEmpty) {
      return RecognitionBranch.failure;
    }

    final candidate = top1OrNull;
    if (candidate == null) {
      return RecognitionBranch.failure;
    }

    if (candidate.confidence < AppConstants.recognitionLowConfidenceThreshold) {
      return RecognitionBranch.failure;
    }
    if (candidate.confidence >= AppConstants.recognitionStrongThreshold) {
      return RecognitionBranch.strong;
    }
    return RecognitionBranch.candidates;
  }

  bool get isConfident => branch == RecognitionBranch.strong;
}
