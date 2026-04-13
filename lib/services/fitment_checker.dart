import '../models/fitment_result.dart';
import '../models/vehicle_fitment.dart';
import '../models/wheel_spec.dart';

class FitmentChecker {
  static const double etSlightThreshold = 3.0;
  static const double etStrongThreshold = 8.0;

  static const double widthSlightThreshold = 0.5;
  static const double widthStrongThreshold = 1.0;

  static const double diameterSlightThreshold = 1.0;
  static const double diameterStrongThreshold = 2.0;

  FitmentResult check(WheelSpec wheel, VehicleFitment vehicle) {
    final reasons = <FitmentReason>[];
    var hasHardFail = false;
    var hasStrongRangeFail = false;

    void addReason(String code, String message, {bool hardFail = false}) {
      reasons.add(FitmentReason(code: code, message: message, isHardFail: hardFail));
      if (hardFail) {
        hasHardFail = true;
      }
    }

    final wheelPcd = wheel.pcd?.trim();
    final vehiclePcd = vehicle.pcd?.trim();
    if (wheelPcd != null && vehiclePcd != null && wheelPcd.isNotEmpty && vehiclePcd.isNotEmpty) {
      if (wheelPcd.toLowerCase() != vehiclePcd.toLowerCase()) {
        addReason(
          'pcd_mismatch',
          'Varžtų išdėstymo neatitikimas: ratlankis $wheelPcd, automobilis $vehiclePcd.',
          hardFail: true,
        );
      }
    } else {
      addReason('pcd_missing', 'Trūksta PCD reikšmės, būtina papildoma patikra.');
    }

    if (wheel.boltCount != null && vehicle.boltCount != null) {
      if (wheel.boltCount != vehicle.boltCount) {
        addReason(
          'bolt_count_mismatch',
          'Varžtų skaičiaus neatitikimas: ratlankis ${wheel.boltCount}, automobilis ${vehicle.boltCount}.',
          hardFail: true,
        );
      }
    } else {
      addReason('bolt_count_missing', 'Trūksta varžtų skaičiaus, būtina papildoma patikra.');
    }

    if (wheel.cb != null && vehicle.cb != null) {
      if (wheel.cb! < vehicle.cb!) {
        addReason(
          'cb_too_small',
          'Ratlankio centrinė skylė per maža: ${wheel.cb} mm, automobilio stebulė ${vehicle.cb} mm.',
          hardFail: true,
        );
      } else if (wheel.cb! > vehicle.cb!) {
        addReason(
          'cb_larger',
          'Ratlankio centrinė skylė didesnė nei stebulė; gali reikėti centravimo žiedų.',
        );
      }
    } else {
      addReason('cb_missing', 'Trūksta centrinės skylės (CB) reikšmės, būtina papildoma patikra.');
    }

    final wheelThread = wheel.threadSize?.trim();
    final vehicleThread = vehicle.threadSize?.trim();
    if (wheelThread != null && vehicleThread != null && wheelThread.isNotEmpty && vehicleThread.isNotEmpty) {
      if (wheelThread.toLowerCase() != vehicleThread.toLowerCase()) {
        addReason(
          'thread_mismatch',
          'Sriegio neatitikimas: ratlankis $wheelThread, automobilis $vehicleThread.',
          hardFail: true,
        );
      }
    }

    final diameterCheck = _checkRange(
      value: wheel.diameterIn,
      min: vehicle.diameterMinIn,
      max: vehicle.diameterMaxIn,
      slightThreshold: diameterSlightThreshold,
      strongThreshold: diameterStrongThreshold,
      insideCode: 'diameter_ok',
      slightCode: 'diameter_slight_out',
      strongCode: 'diameter_strong_out',
      fieldNameLt: 'Ratlankio diametras',
      unit: 'col.',
    );
    if (diameterCheck != null) {
      reasons.add(diameterCheck);
      if (diameterCheck.code == 'diameter_strong_out') {
        hasStrongRangeFail = true;
      }
    }

    final widthCheck = _checkRange(
      value: wheel.widthJ,
      min: vehicle.widthMinJ,
      max: vehicle.widthMaxJ,
      slightThreshold: widthSlightThreshold,
      strongThreshold: widthStrongThreshold,
      insideCode: 'width_ok',
      slightCode: 'width_slight_out',
      strongCode: 'width_strong_out',
      fieldNameLt: 'Ratlankio plotis',
      unit: 'J',
    );
    if (widthCheck != null) {
      reasons.add(widthCheck);
      if (widthCheck.code == 'width_strong_out') {
        hasStrongRangeFail = true;
      }
    }

    final etCheck = _checkRange(
      value: wheel.et,
      min: vehicle.etMin,
      max: vehicle.etMax,
      slightThreshold: etSlightThreshold,
      strongThreshold: etStrongThreshold,
      insideCode: 'et_ok',
      slightCode: 'et_slight_out',
      strongCode: 'et_strong_out',
      fieldNameLt: 'Išnešimas ET',
      unit: '',
    );
    if (etCheck != null) {
      reasons.add(etCheck);
      if (etCheck.code == 'et_strong_out') {
        hasStrongRangeFail = true;
      }
    }

    final status = hasHardFail || hasStrongRangeFail
        ? FitmentStatus.incompatible
        : reasons.isNotEmpty
            ? FitmentStatus.caution
            : FitmentStatus.compatible;

    return FitmentResult(status: status, reasons: reasons);
  }

  FitmentReason? _checkRange({
    required double? value,
    required double? min,
    required double? max,
    required double slightThreshold,
    required double strongThreshold,
    required String insideCode,
    required String slightCode,
    required String strongCode,
    required String fieldNameLt,
    required String unit,
  }) {
    if (value == null || min == null || max == null) {
      return FitmentReason(
        code: '${insideCode}_missing',
        message: '$fieldNameLt: trūksta duomenų, reikalinga papildoma patikra.',
      );
    }

    if (value >= min && value <= max) {
      return null;
    }

    final deviation = value < min ? min - value : value - max;
    final unitSuffix = unit.isEmpty ? '' : ' $unit';

    if (deviation <= slightThreshold) {
      return FitmentReason(
        code: slightCode,
        message:
            '$fieldNameLt šiek tiek už OEM ribų: ratlankis $value$unitSuffix, leistina $min-$max$unitSuffix.',
      );
    }

    if (deviation > strongThreshold) {
      return FitmentReason(
        code: strongCode,
        message:
            '$fieldNameLt ženkliai neatitinka OEM ribų: ratlankis $value$unitSuffix, leistina $min-$max$unitSuffix.',
        isHardFail: true,
      );
    }

    return FitmentReason(
      code: slightCode,
      message:
          '$fieldNameLt už OEM ribų: ratlankis $value$unitSuffix, leistina $min-$max$unitSuffix.',
    );
  }
}
