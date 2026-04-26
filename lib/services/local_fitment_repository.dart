import '../models/vehicle_fitment.dart';
import '../models/wheel_spec.dart';
import 'fitment_database.dart';

class LocalFitmentRepository {
  final FitmentDatabase _database;

  LocalFitmentRepository({FitmentDatabase? database})
    : _database = database ?? FitmentDatabase.instance;

  List<WheelSpec> _mergeFinishOnlyDuplicates(List<WheelSpec> wheels) {
    final grouped = <String, WheelSpec>{};

    for (final wheel in wheels) {
      final key = wheel.technicalKey;
      final existing = grouped[key];
      if (existing == null) {
        grouped[key] = wheel;
      } else {
        grouped[key] = existing.mergeFinishOnly(wheel);
      }
    }

    return grouped.values.toList(growable: false);
  }

  Future<List<WheelSpec>> loadWheelSpecs() async {
    final rows = await _database.loadWheelModels();

    final rawSpecs = rows
        .map((row) => Map<String, dynamic>.from(row))
        .map(WheelSpec.fromJson)
        .where((w) => w.wheelClassName.trim().isNotEmpty)
        .toList(growable: false);

    return _mergeFinishOnlyDuplicates(rawSpecs);
  }

  Future<List<VehicleFitment>> loadVehicleFitments() async {
    final rows = await _database.loadCarModels();

    return rows
        .map((row) => Map<String, dynamic>.from(row))
        .map(VehicleFitment.fromJson)
        .where((v) => (v.generation ?? '').trim().isNotEmpty)
        .toList(growable: false);
  }

  WheelSpec? findByPredictedClass(
    List<WheelSpec> wheelSpecs,
    String predictedLabel,
  ) {
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
