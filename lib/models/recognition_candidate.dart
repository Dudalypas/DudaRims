class RecognitionCandidate {
  final String label;
  final double score;

  const RecognitionCandidate({required this.label, required this.score});

  double get confidence => score;
}
