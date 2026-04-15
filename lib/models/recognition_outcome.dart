import 'recognition_candidate.dart';

class RecognitionOutcome {
  final List<RecognitionCandidate> top5;
  final bool isConfident;

  const RecognitionOutcome({
    required this.top5,
    required this.isConfident,
  });

  RecognitionCandidate get top1 => top5.first;
  List<RecognitionCandidate> get top3 => top5.take(3).toList(growable: false);
}
