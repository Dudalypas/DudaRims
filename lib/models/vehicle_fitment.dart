class VehicleFitment {
  final String? brand;
  final String? model;
  final String? generation;
  final int? yearFrom;
  final int? yearTo;
  final String? pcd;
  final double? cb;
  final int? boltCount;
  final String? threadSize;
  final double? diameterMinIn;
  final double? diameterMaxIn;
  final double? widthMinJ;
  final double? widthMaxJ;
  final double? etMin;
  final double? etMax;

  const VehicleFitment({
    this.brand,
    this.model,
    this.generation,
    this.yearFrom,
    this.yearTo,
    this.pcd,
    this.cb,
    this.boltCount,
    this.threadSize,
    this.diameterMinIn,
    this.diameterMaxIn,
    this.widthMinJ,
    this.widthMaxJ,
    this.etMin,
    this.etMax,
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

  factory VehicleFitment.fromJson(Map<String, dynamic> json) {
    return VehicleFitment(
      brand: _asString(json['brand']),
      model: _asString(json['model']),
      generation: _asString(json['generation']),
      yearFrom: _asInt(json['year_from']),
      yearTo: _asInt(json['year_to']),
      pcd: _asString(json['pcd']),
      cb: _asDouble(json['cb']),
      boltCount: _asInt(json['bolt_count']),
      threadSize: _asString(json['thread_size']),
      diameterMinIn: _asDouble(json['diameter_min_in']),
      diameterMaxIn: _asDouble(json['diameter_max_in']),
      widthMinJ: _asDouble(json['width_min_j']),
      widthMaxJ: _asDouble(json['width_max_j']),
      etMin: _asDouble(json['et_min']),
      etMax: _asDouble(json['et_max']),
    );
  }

  String get generationLabel {
    final gen = (generation ?? '-').trim();
    if (yearFrom != null && yearTo != null) {
      return '$gen ($yearFrom-$yearTo)';
    }
    return gen;
  }

  Map<String, dynamic> toJson() {
    return {
      'brand': brand,
      'model': model,
      'generation': generation,
      'year_from': yearFrom,
      'year_to': yearTo,
      'pcd': pcd,
      'cb': cb,
      'bolt_count': boltCount,
      'thread_size': threadSize,
      'diameter_min_in': diameterMinIn,
      'diameter_max_in': diameterMaxIn,
      'width_min_j': widthMinJ,
      'width_max_j': widthMaxJ,
      'et_min': etMin,
      'et_max': etMax,
    };
  }
}
