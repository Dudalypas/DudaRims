import 'dart:convert';

import 'package:flutter/services.dart';

import '../models/vehicle_fitment.dart';
import '../models/wheel_spec.dart';

class LocalFitmentRepository {
  // Hook point: swap these local assets with a database/API adapter in the future.
  static const _wheelsPath = 'assets/data/wheels.json';
  static const _vehicleFitmentPath = 'assets/data/vehicle_fitment.json';
  static const _vehicleFitmentFallbackPath = 'assets/data/skoda_models.json';

  Future<List<WheelSpec>> loadWheelSpecs() async {
    final raw = await rootBundle.loadString(_wheelsPath);
    final decoded = jsonDecode(_sanitizeJson(raw));
    if (decoded is! List) {
      throw const FormatException('wheels.json format is invalid');
    }

    return decoded
        .whereType<Map>()
        .map((e) => Map<String, dynamic>.from(e))
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

    final decoded = jsonDecode(_sanitizeJson(raw));
    if (decoded is! List) {
      throw const FormatException('vehicle_fitment.json format is invalid');
    }

    return decoded
        .whereType<Map>()
        .map((e) => Map<String, dynamic>.from(e))
        .map(VehicleFitment.fromJson)
        .where((v) => (v.generation ?? '').trim().isNotEmpty)
        .toList(growable: false);
  }

  String _sanitizeJson(String value) {
    // Some generated datasets may contain NaN tokens from pandas exports.
    return value.replaceAll(RegExp(r'\bNaN\b'), 'null');
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
