class RetrievalResult {
  final String label;
  final double cosineSimilarity;

  const RetrievalResult({required this.label, required this.cosineSimilarity});

  double get normalizedScore {
    final normalized = (cosineSimilarity + 1.0) / 2.0;
    if (normalized < 0.0) return 0.0;
    if (normalized > 1.0) return 1.0;
    return normalized;
  }
}
