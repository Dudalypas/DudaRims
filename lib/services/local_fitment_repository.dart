import 'dart:convert';

import 'package:flutter/services.dart';

import '../models/vehicle_fitment.dart';
import '../models/wheel_spec.dart';

class LocalFitmentRepository {
  static const _wheelsPath = 'assets/data/wheels.json';
  static const _vehicleFitmentPath = 'assets/data/vehicle_fitment.json';
  static const _vehicleFitmentFallbackPath = 'assets/data/skoda_models.json';

  List<dynamic> _decodeList(String raw, String sourceName) {
    final normalized = raw
        .replaceAll(RegExp(r'(?<=[:\[,\s])NaN(?=[,\]\s}])'), 'null')
        .replaceAll(RegExp(r'(?<=[:\[,\s])Infinity(?=[,\]\s}])'), 'null')
        .replaceAll(RegExp(r'(?<=[:\[,\s])-Infinity(?=[,\]\s}])'), 'null');

    final decoded = jsonDecode(normalized);
    if (decoded is! List) {
      throw FormatException('$sourceName format is invalid');
    }
    return decoded;
  }

  Future<List<WheelSpec>> loadWheelSpecs() async {
    final raw = await rootBundle.loadString(_wheelsPath);
    final decoded = _decodeList(raw, 'wheels.json');

    return decoded
        .whereType<Map<String, dynamic>>()
        .map(WheelSpec.fromJson)
        .where((w) => w.wheelClassName.trim().isNotEmpty)
        .toList(growable: false);
  }

  Future<List<VehicleFitment>> loadVehicleFitments() async {
    String raw;
    try {
      raw = await rootBundle.loadString(_vehicleFitmentPath);
    } catch (_) {
      raw = await rootBundle.loadString(_vehicleFitmentFallbackPath);
    }
    final decoded = _decodeList(raw, 'vehicle_fitment.json');

    return decoded
        .whereType<Map<String, dynamic>>()
        .map(VehicleFitment.fromJson)
        .where((v) => (v.generation ?? '').trim().isNotEmpty)
        .toList(growable: false);
  }

  WheelSpec? findByPredictedClass(List<WheelSpec> wheelSpecs, String predictedLabel) {
    final target = predictedLabel.trim().toLowerCase();
    if (target.isEmpty) return null;

    for (final spec in wheelSpecs) {
      if (spec.wheelClassName.trim().toLowerCase() == target) {
        return spec;
      }
    }
    return null;
  }
}
