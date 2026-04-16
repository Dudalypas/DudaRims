import '../core/constants/app_constants.dart';
import '../models/fitment_result.dart';
import '../models/vehicle_fitment.dart';
import '../models/wheel_spec.dart';

class FitmentChecker {
  FitmentResult check(WheelSpec wheel, VehicleFitment vehicle) {
    final checks = <FitmentParameterCheck>[
      _evaluatePcd(wheel, vehicle),
      _evaluateCb(wheel, vehicle),
      _evaluateRange(
        parameter: FitmentParameter.diameter,
        value: wheel.diameterIn,
        min: vehicle.diameterMinIn,
        max: vehicle.diameterMaxIn,
        warningDeviation: AppConstants.fitmentDiameterWarningDeviationIn,
        unit: 'col.',
        label: 'Skersmuo',
      ),
      _evaluateRange(
        parameter: FitmentParameter.width,
        value: wheel.widthJ,
        min: vehicle.widthMinJ,
        max: vehicle.widthMaxJ,
        warningDeviation: AppConstants.fitmentWidthWarningDeviationJ,
        unit: 'J',
        label: 'Plotis',
      ),
      _evaluateRange(
        parameter: FitmentParameter.et,
        value: wheel.et,
        min: vehicle.etMin,
        max: vehicle.etMax,
        warningDeviation: AppConstants.fitmentEtWarningDeviationMm,
        unit: '',
        label: 'ET',
      ),
    ];

    final hasFail = checks.any((c) => c.status == FitmentParameterStatus.fail);
    final hasWarning = checks.any(
      (c) => c.status == FitmentParameterStatus.warning,
    );

    final status = hasFail
        ? FitmentStatus.incompatible
        : hasWarning
        ? FitmentStatus.caution
        : FitmentStatus.compatible;

    final orderedChecks = List<FitmentParameterCheck>.from(checks)
      ..sort((a, b) {
        final severity = _severityRank(
          a.status,
        ).compareTo(_severityRank(b.status));
        if (severity != 0) return severity;
        final critical = (b.isCritical ? 1 : 0).compareTo(a.isCritical ? 1 : 0);
        if (critical != 0) return critical;
        return a.parameter.index.compareTo(b.parameter.index);
      });

    return FitmentResult(status: status, checks: orderedChecks);
  }

  int _severityRank(FitmentParameterStatus status) {
    switch (status) {
      case FitmentParameterStatus.fail:
        return 0;
      case FitmentParameterStatus.warning:
        return 1;
      case FitmentParameterStatus.ok:
        return 2;
    }
  }

  FitmentParameterCheck _evaluatePcd(WheelSpec wheel, VehicleFitment vehicle) {
    final wheelPcd = wheel.pcd?.trim();
    final carPcd = vehicle.pcd?.trim();

    if (wheelPcd == null ||
        carPcd == null ||
        wheelPcd.isEmpty ||
        carPcd.isEmpty) {
      return const FitmentParameterCheck(
        parameter: FitmentParameter.pcd,
        rimValue: '-',
        expectedValue: '-',
        status: FitmentParameterStatus.warning,
        message: 'PCD reikšmės trūksta, reikalinga papildoma patikra.',
        isCritical: true,
      );
    }

    final matches = wheelPcd.toLowerCase() == carPcd.toLowerCase();
    return FitmentParameterCheck(
      parameter: FitmentParameter.pcd,
      rimValue: wheelPcd,
      expectedValue: carPcd,
      status: matches ? FitmentParameterStatus.ok : FitmentParameterStatus.fail,
      message: matches
          ? 'PCD sutampa.'
          : 'PCD nesutampa: ratlankis $wheelPcd, automobiliui reikalinga $carPcd.',
      isCritical: true,
    );
  }

  FitmentParameterCheck _evaluateCb(WheelSpec wheel, VehicleFitment vehicle) {
    final rimCb = wheel.cb;
    final carCb = vehicle.cb;
    if (rimCb == null || carCb == null) {
      return const FitmentParameterCheck(
        parameter: FitmentParameter.cb,
        rimValue: '-',
        expectedValue: '-',
        status: FitmentParameterStatus.warning,
        message: 'CB reikšmės trūksta, reikalinga papildoma patikra.',
        isCritical: true,
      );
    }

    final ok = rimCb >= carCb;
    return FitmentParameterCheck(
      parameter: FitmentParameter.cb,
      rimValue: _formatNum(rimCb),
      expectedValue: '≥ ${_formatNum(carCb)}',
      status: ok ? FitmentParameterStatus.ok : FitmentParameterStatus.fail,
      message: ok
          ? 'CB sutampa.'
          : 'CB per mažas: ${_formatNum(rimCb)} mm, reikalinga bent ${_formatNum(carCb)} mm.',
      isCritical: true,
    );
  }

  FitmentParameterCheck _evaluateRange({
    required FitmentParameter parameter,
    required double? value,
    required double? min,
    required double? max,
    required double warningDeviation,
    required String label,
    required String unit,
  }) {
    if (value == null || min == null || max == null) {
      return FitmentParameterCheck(
        parameter: parameter,
        rimValue: '-',
        expectedValue: '-',
        status: FitmentParameterStatus.warning,
        message: '$label reikšmės trūksta, reikalinga papildoma patikra.',
      );
    }

    final rangeText =
        '${_formatNum(min)}–${_formatNum(max)}${unit.isEmpty ? '' : ' $unit'}';
    final rimText = '${_formatNum(value)}${unit.isEmpty ? '' : ' $unit'}';

    if (value >= min && value <= max) {
      return FitmentParameterCheck(
        parameter: parameter,
        rimValue: _formatNum(value),
        expectedValue: rangeText,
        status: FitmentParameterStatus.ok,
        message: '$label atitinka leistiną intervalą.',
      );
    }

    final deviation = value < min ? min - value : value - max;
    final isHigh = value > max;

    if (deviation <= warningDeviation) {
      return FitmentParameterCheck(
        parameter: parameter,
        rimValue: _formatNum(value),
        expectedValue: rangeText,
        status: FitmentParameterStatus.warning,
        message: isHigh
            ? '$label šiek tiek viršija rekomenduojamą ribą: $rimText, leidžiama iki ${_formatNum(max)}${unit.isEmpty ? '' : ' $unit'}.'
            : '$label šiek tiek mažesnis nei rekomenduojama: $rimText, leidžiama nuo ${_formatNum(min)}${unit.isEmpty ? '' : ' $unit'}.',
      );
    }

    return FitmentParameterCheck(
      parameter: parameter,
      rimValue: _formatNum(value),
      expectedValue: rangeText,
      status: FitmentParameterStatus.fail,
      message: isHigh
          ? '$label per didelis: $rimText, kai automobiliui leidžiama $rangeText.'
          : '$label per mažas: $rimText, kai automobiliui leidžiama $rangeText.',
    );
  }

  String _formatNum(double value) {
    return value % 1 == 0 ? value.toStringAsFixed(0) : value.toStringAsFixed(1);
  }
}
