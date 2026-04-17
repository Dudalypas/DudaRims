String? resolveRimThumbnailAsset(String classLabel) {
  final key = classLabel  
      .trim()
      .toLowerCase()
      .replaceAll(' ', '_');

  if (key.isEmpty) return null;

  return 'assets/data/rim_thumbnails/$key.jpg';
}

