const Map<String, String> rimThumbnailByClass = {
  // Add official/downloaded thumbnails here.
  // Key format: lowercased class label, e.g. "braga_diamond_cut_black"
  // Value format: local asset path, e.g. "assets/data/rim_thumbnails/braga_diamond_cut_black.webp"
};

String? resolveRimThumbnailAsset(String classLabel) {
  final key = classLabel.trim().toLowerCase();
  if (key.isEmpty) return null;

  final mapped = rimThumbnailByClass[key];
  if (mapped != null && mapped.isNotEmpty) {
    return mapped;
  }
  
  // Convention fallback when map entry is not provided.
  return 'assets/data/rim_thumbnails/$key.jpg';
}
