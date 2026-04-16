import 'package:flutter_test/flutter_test.dart';

import 'package:dudarims/models/fitment_result.dart';
import 'package:dudarims/models/vehicle_fitment.dart';
import 'package:dudarims/models/wheel_spec.dart';
import 'package:dudarims/services/fitment_checker.dart';

void main() {
  final checker = FitmentChecker();

  VehicleFitment octaviaMk2() {
    return const VehicleFitment(
      brand: 'Skoda',
      model: 'Octavia',
      generation: 'Mk2 A5',
      pcd: '5x112',
      cb: 57.1,
      diameterMinIn: 15.0,
      diameterMaxIn: 18.0,
      widthMinJ: 6.0,
      widthMaxJ: 7.5,
      etMin: 47.0,
      etMax: 54.0,
    );
  }

  WheelSpec wheel({
    double diameter = 17,
    double width = 7.0,
    double et = 50,
    String pcd = '5x112',
    double cb = 57.1,
  }) {
    return WheelSpec(
      wheelClassName: 'TestWheel',
      diameterIn: diameter,
      widthJ: width,
      et: et,
      pcd: pcd,
      cb: cb,
    );
  }

  group('FitmentChecker', () {
    test('perfect fit => Tinka', () {
      final result = checker.check(wheel(), octaviaMk2());
      expect(result.status, FitmentStatus.compatible);
      expect(
        result.checks.every((c) => c.status == FitmentParameterStatus.ok),
        isTrue,
      );
    });

    test('only small ET deviation => warning', () {
      final result = checker.check(wheel(et: 45), octaviaMk2());
      expect(result.status, FitmentStatus.caution);
      final etCheck = result.checks.firstWhere(
        (c) => c.parameter == FitmentParameter.et,
      );
      expect(etCheck.status, FitmentParameterStatus.warning);
    });

    test('only small width deviation => warning', () {
      final result = checker.check(wheel(width: 7.9), octaviaMk2());
      expect(result.status, FitmentStatus.caution);
      final widthCheck = result.checks.firstWhere(
        (c) => c.parameter == FitmentParameter.width,
      );
      expect(widthCheck.status, FitmentParameterStatus.warning);
    });

    test('diameter too large => fail', () {
      final result = checker.check(wheel(diameter: 20), octaviaMk2());
      expect(result.status, FitmentStatus.incompatible);
      final diameterCheck = result.checks.firstWhere(
        (c) => c.parameter == FitmentParameter.diameter,
      );
      expect(diameterCheck.status, FitmentParameterStatus.fail);
    });

    test('PCD mismatch => fail', () {
      final result = checker.check(wheel(pcd: '5x100'), octaviaMk2());
      expect(result.status, FitmentStatus.incompatible);
      final pcdCheck = result.checks.firstWhere(
        (c) => c.parameter == FitmentParameter.pcd,
      );
      expect(pcdCheck.status, FitmentParameterStatus.fail);
    });

    test('CB mismatch (too small) => fail', () {
      final result = checker.check(wheel(cb: 56.0), octaviaMk2());
      expect(result.status, FitmentStatus.incompatible);
      final cbCheck = result.checks.firstWhere(
        (c) => c.parameter == FitmentParameter.cb,
      );
      expect(cbCheck.status, FitmentParameterStatus.fail);
    });

    test('multiple warnings but no fails => fits with considerations', () {
      final result = checker.check(wheel(width: 7.9, et: 45), octaviaMk2());
      expect(result.status, FitmentStatus.caution);
      expect(
        result.checks.any((c) => c.status == FitmentParameterStatus.fail),
        isFalse,
      );
      expect(
        result.checks
            .where((c) => c.status == FitmentParameterStatus.warning)
            .length,
        greaterThanOrEqualTo(2),
      );
    });

    test('Xtreme 20", 8J, ET41 against Octavia Mk2 limits is incompatible', () {
      final xtreme = wheel(diameter: 20, width: 8.0, et: 41);
      final result = checker.check(xtreme, octaviaMk2());

      expect(result.status, FitmentStatus.incompatible);

      final diameterCheck = result.checks.firstWhere(
        (c) => c.parameter == FitmentParameter.diameter,
      );
      final widthCheck = result.checks.firstWhere(
        (c) => c.parameter == FitmentParameter.width,
      );
      final etCheck = result.checks.firstWhere(
        (c) => c.parameter == FitmentParameter.et,
      );

      expect(diameterCheck.status, FitmentParameterStatus.fail);
      expect(widthCheck.status, FitmentParameterStatus.warning);
      expect(etCheck.status, FitmentParameterStatus.fail);
    });
  });
}
