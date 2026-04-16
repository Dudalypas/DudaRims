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
  final List<String> availableColorVariants;

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
    this.availableColorVariants = const [],
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

  static List<String> _asStringList(dynamic value) {
    if (value == null) return const [];
    if (value is List) {
      return value
          .map((e) => e?.toString().trim() ?? '')
          .where((e) => e.isNotEmpty)
          .toSet()
          .toList(growable: false);
    }
    final single = _asString(value);
    return single == null ? const [] : [single];
  }

  static String _numKey(double? value) {
    if (value == null) return '-';
    return value.toStringAsFixed(2);
  }

  String get technicalKey {
    return [
      wheelClassName.toLowerCase(),
      _numKey(diameterIn),
      _numKey(widthJ),
      _numKey(et),
      (pcd ?? '-').toLowerCase(),
      _numKey(cb),
      boltCount?.toString() ?? '-',
      (threadSize ?? '-').toLowerCase(),
    ].join('|');
  }

  List<String> get finishes {
    final out = <String>{
      ...availableColorVariants.where((f) => f.trim().isNotEmpty),
      if ((colorVariant ?? '').trim().isNotEmpty) colorVariant!.trim(),
    };
    return out.toList(growable: false);
  }

  WheelSpec mergeFinishOnly(WheelSpec other) {
    final mergedFinishes = <String>{
      ...finishes,
      ...other.finishes,
    }.where((f) => f.trim().isNotEmpty).toList(growable: false);

    return WheelSpec(
      wheelClassName: wheelClassName,
      officialWheelName: officialWheelName ?? other.officialWheelName,
      brand: brand ?? other.brand,
      colorVariant: mergedFinishes.isEmpty ? null : mergedFinishes.first,
      diameterIn: diameterIn ?? other.diameterIn,
      widthJ: widthJ ?? other.widthJ,
      et: et ?? other.et,
      pcd: pcd ?? other.pcd,
      cb: cb ?? other.cb,
      boltCount: boltCount ?? other.boltCount,
      threadSize: threadSize ?? other.threadSize,
      designedForModels: designedForModels ?? other.designedForModels,
      availableColorVariants: mergedFinishes,
    );
  }

  factory WheelSpec.fromJson(Map<String, dynamic> json) {
    final singleColor = _asString(json['color_variant']);
    final allColors = _asStringList(json['available_color_variants']);
    final mergedColors = <String>{
      ...allColors,
      ?singleColor,
    }.toList(growable: false);

    return WheelSpec(
      wheelClassName: (json['wheel_class_name'] ?? '').toString().trim(),
      officialWheelName: _asString(json['official_wheel_name']),
      brand: _asString(json['brand']),
      colorVariant: singleColor,
      diameterIn: _asDouble(json['diameter_in']),
      widthJ: _asDouble(json['width_j']),
      et: _asDouble(json['et']),
      pcd: _asString(json['pcd']),
      cb: _asDouble(json['cb']),
      boltCount: _asInt(json['bolt_count']),
      threadSize: _asString(json['thread_size']),
      designedForModels: _asString(json['designed_for_models']),
      availableColorVariants: mergedColors,
    );
  }
}
