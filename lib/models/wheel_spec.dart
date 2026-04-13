class WheelSpec {
  final String wheelClassName;
  final String? officialWheelName;
  final String? brand;
  final String? colorVariant;
  final double? diameterIn;
  final double? widthJ;
  final double? et;
  final String? pcd;
  final double? cb;
  final int? boltCount;
  final String? threadSize;
  final String? designedForModels;

  const WheelSpec({
    required this.wheelClassName,
    this.officialWheelName,
    this.brand,
    this.colorVariant,
    this.diameterIn,
    this.widthJ,
    this.et,
    this.pcd,
    this.cb,
    this.boltCount,
    this.threadSize,
    this.designedForModels,
  });

  static double? _asDouble(dynamic value) {
    if (value == null) return null;
    if (value is num) return value.toDouble();
    final text = value.toString().trim().replaceAll(',', '.');
    if (text.isEmpty) return null;
    return double.tryParse(text);
  }

  static int? _asInt(dynamic value) {
    if (value == null) return null;
    if (value is int) return value;
    if (value is num) return value.toInt();
    final text = value.toString().trim();
    if (text.isEmpty) return null;
    return int.tryParse(text);
  }

  static String? _asString(dynamic value) {
    if (value == null) return null;
    final text = value.toString().trim();
    return text.isEmpty ? null : text;
  }

  factory WheelSpec.fromJson(Map<String, dynamic> json) {
    return WheelSpec(
      wheelClassName: (json['wheel_class_name'] ?? '').toString().trim(),
      officialWheelName: _asString(json['official_wheel_name']),
      brand: _asString(json['brand']),
      colorVariant: _asString(json['color_variant']),
      diameterIn: _asDouble(json['diameter_in']),
      widthJ: _asDouble(json['width_j']),
      et: _asDouble(json['et']),
      pcd: _asString(json['pcd']),
      cb: _asDouble(json['cb']),
      boltCount: _asInt(json['bolt_count']),
      threadSize: _asString(json['thread_size']),
      designedForModels: _asString(json['designed_for_models']),
    );
  }
}
