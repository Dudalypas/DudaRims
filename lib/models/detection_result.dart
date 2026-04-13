class DetectionResult {
  final double left;
  final double top;
  final double right;
  final double bottom;
  final double score;

  const DetectionResult({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
    required this.score,
  });

  double get width => right - left;
  double get height => bottom - top;
}
