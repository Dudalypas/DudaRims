class RetrievalReference {
  final String label;
  final List<List<double>> embeddings;
  final List<double>? centroidEmbedding;
  final int sampleCount;

  const RetrievalReference({
    required this.label,
    required this.embeddings,
    this.centroidEmbedding,
    required this.sampleCount,
  });
}
