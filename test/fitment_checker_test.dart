import 'package:flutter_test/flutter_test.dart';

import 'package:dudarims/models/fitment_result.dart';
import 'package:dudarims/models/vehicle_fitment.dart';
import 'package:dudarims/models/wheel_spec.dart';
import 'package:dudarims/services/fitment_checker.dart';

void main() {
  // Referencinis automobilis testams (Škoda Octavia Mk2)
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

  // Numatytas rato konstruktorius (puikiai suderintas)
  WheelSpec wheel({
    double diameter = 17.0,
    double width = 7.0,
    double et = 50.0,
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

  group('Tinkamumo tikrintuvas', () {
    late FitmentChecker checker;

    setUp(() {
      checker = FitmentChecker();
    });

    group('Puikus suderinamumas', () {
      test('visi parametrai intervale => suderinama', () {
        final result = checker.check(wheel(), octaviaMk2());
        expect(result.status, FitmentStatus.compatible);
        expect(
          result.checks.every((c) => c.status == FitmentParameterStatus.ok),
          isTrue,
        );
      });
    });

    group('PCD tikrinimas', () {
      test('PCD sutampa => gerai', () {
        final result = checker.check(wheel(pcd: '5x112'), octaviaMk2());
        final pcdCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.pcd,
        );
        expect(pcdCheck.status, FitmentParameterStatus.ok);
      });

      test('PCD sutampa (nesvarbus didž./maž.) => gerai', () {
        final result = checker.check(wheel(pcd: '5X112'), octaviaMk2());
        final pcdCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.pcd,
        );
        expect(pcdCheck.status, FitmentParameterStatus.ok);
      });

      test('PCD neatitinka => nesuderinama', () {
        final result = checker.check(wheel(pcd: '5x100'), octaviaMk2());
        expect(result.status, FitmentStatus.incompatible);
        final pcdCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.pcd,
        );
        expect(pcdCheck.status, FitmentParameterStatus.fail);
      });
    });

    group('CB (centrinis anga) tikrinimas', () {
      test('CB = autom. min => gerai', () {
        final result = checker.check(wheel(cb: 57.1), octaviaMk2());
        final cbCheck =
            result.checks.firstWhere((c) => c.parameter == FitmentParameter.cb);
        expect(cbCheck.status, FitmentParameterStatus.ok);
      });

      test('CB didesnis už autom. min => gerai', () {
        final result = checker.check(wheel(cb: 66.6), octaviaMk2());
        final cbCheck =
            result.checks.firstWhere((c) => c.parameter == FitmentParameter.cb);
        expect(cbCheck.status, FitmentParameterStatus.ok);
      });

      test('CB mažesnis už autom. min => nesuderinama', () {
        final result = checker.check(wheel(cb: 56.0), octaviaMk2());
        expect(result.status, FitmentStatus.incompatible);
        final cbCheck =
            result.checks.firstWhere((c) => c.parameter == FitmentParameter.cb);
        expect(cbCheck.status, FitmentParameterStatus.fail);
      });

      test('CB gerokai mažesnis už autom. min => nesuderinama', () {
        final result = checker.check(wheel(cb: 50.0), octaviaMk2());
        final cbCheck =
            result.checks.firstWhere((c) => c.parameter == FitmentParameter.cb);
        expect(cbCheck.status, FitmentParameterStatus.fail);
      });
    });

    group('Skersmens tikrinimas (min: 15.0, max: 18.0, įspėjimo nuokrypis: 1.0)', () {
      test('skersmuo = minimumas => gerai', () {
        final result = checker.check(wheel(diameter: 15.0), octaviaMk2());
        final diamCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.diameter,
        );
        expect(diamCheck.status, FitmentParameterStatus.ok);
      });

      test('skersmuo = maksimumas => gerai', () {
        final result = checker.check(wheel(diameter: 18.0), octaviaMk2());
        final diamCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.diameter,
        );
        expect(diamCheck.status, FitmentParameterStatus.ok);
      });

      test('skersmuo šiek tiek žemiau minimumo (įspėjimas) => įspėjimas', () {
        final result = checker.check(wheel(diameter: 14.2), octaviaMk2());
        final diamCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.diameter,
        );
        expect(diamCheck.status, FitmentParameterStatus.warning);
        expect(result.status, FitmentStatus.caution);
      });

      test('skersmuo šiek tiek virš maksimumo (įspėjimas) => įspėjimas', () {
        final result = checker.check(wheel(diameter: 18.8), octaviaMk2());
        final diamCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.diameter,
        );
        expect(diamCheck.status, FitmentParameterStatus.warning);
        expect(result.status, FitmentStatus.caution);
      });

      test('gerokai žemiau minimumo => nesuderinama', () {
        final result = checker.check(wheel(diameter: 13.5), octaviaMk2());
        expect(result.status, FitmentStatus.incompatible);
        final diamCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.diameter,
        );
        expect(diamCheck.status, FitmentParameterStatus.fail);
      });

      test('gerokai virš maksimumo => nesuderinama', () {
        final result = checker.check(wheel(diameter: 20.0), octaviaMk2());
        expect(result.status, FitmentStatus.incompatible);
        final diamCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.diameter,
        );
        expect(diamCheck.status, FitmentParameterStatus.fail);
      });
    });

    group('Plotis (min: 6.0, max: 7.5, įsp. nuokrypis: 0.5)', () {
      test('plotis = minimumas => gerai', () {
        final result = checker.check(wheel(width: 6.0), octaviaMk2());
        final widthCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.width,
        );
        expect(widthCheck.status, FitmentParameterStatus.ok);
      });

      test('plotis = maksimumas => gerai', () {
        final result = checker.check(wheel(width: 7.5), octaviaMk2());
        final widthCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.width,
        );
        expect(widthCheck.status, FitmentParameterStatus.ok);
      });

      test('plotis šiek tiek žemiau minimumo (įspėjimas) => įspėjimas', () {
        final result = checker.check(wheel(width: 5.6), octaviaMk2());
        final widthCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.width,
        );
        expect(widthCheck.status, FitmentParameterStatus.warning);
        expect(result.status, FitmentStatus.caution);
      });

      test('plotis šiek tiek virš maksimumo (įspėjimas) => įspėjimas', () {
        final result = checker.check(wheel(width: 7.9), octaviaMk2());
        final widthCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.width,
        );
        expect(widthCheck.status, FitmentParameterStatus.warning);
        expect(result.status, FitmentStatus.caution);
      });

      test('gerokai žemiau minimumo => nesuderinama', () {
        final result = checker.check(wheel(width: 5.0), octaviaMk2());
        expect(result.status, FitmentStatus.incompatible);
        final widthCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.width,
        );
        expect(widthCheck.status, FitmentParameterStatus.fail);
      });

      test('gerokai virš maksimumo => nesuderinama', () {
        final result = checker.check(wheel(width: 9.0), octaviaMk2());
        expect(result.status, FitmentStatus.incompatible);
        final widthCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.width,
        );
        expect(widthCheck.status, FitmentParameterStatus.fail);
      });
    });

    group('ET (min: 47.0, max: 54.0, įsp. nuokrypis: 5.0)', () {
      test('ET = minimumas => gerai', () {
        final result = checker.check(wheel(et: 47.0), octaviaMk2());
        final etCheck =
            result.checks.firstWhere((c) => c.parameter == FitmentParameter.et);
        expect(etCheck.status, FitmentParameterStatus.ok);
      });

      test('ET = maksimumas => gerai', () {
        final result = checker.check(wheel(et: 54.0), octaviaMk2());
        final etCheck =
            result.checks.firstWhere((c) => c.parameter == FitmentParameter.et);
        expect(etCheck.status, FitmentParameterStatus.ok);
      });

      test('ET šiek tiek žemiau minimumo (įspėjimas) => įspėjimas', () {
        final result = checker.check(wheel(et: 43.0), octaviaMk2());
        final etCheck =
            result.checks.firstWhere((c) => c.parameter == FitmentParameter.et);
        expect(etCheck.status, FitmentParameterStatus.warning);
        expect(result.status, FitmentStatus.caution);
      });

      test('ET šiek tiek virš maksimumo (įspėjimas) => įspėjimas', () {
        final result = checker.check(wheel(et: 58.5), octaviaMk2());
        final etCheck =
            result.checks.firstWhere((c) => c.parameter == FitmentParameter.et);
        expect(etCheck.status, FitmentParameterStatus.warning);
        expect(result.status, FitmentStatus.caution);
      });

      test('gerokai žemiau minimumo => nesuderinama', () {
        final result = checker.check(wheel(et: 35.0), octaviaMk2());
        expect(result.status, FitmentStatus.incompatible);
        final etCheck =
            result.checks.firstWhere((c) => c.parameter == FitmentParameter.et);
        expect(etCheck.status, FitmentParameterStatus.fail);
      });

      test('gerokai virš maksimumo => nesuderinama', () {
        final result = checker.check(wheel(et: 65.0), octaviaMk2());
        expect(result.status, FitmentStatus.incompatible);
        final etCheck =
            result.checks.firstWhere((c) => c.parameter == FitmentParameter.et);
        expect(etCheck.status, FitmentParameterStatus.fail);
      });
    });

    group('Kelių parametrų scenarijai', () {
      test('kelios įspėjimai, nėra klaidų => atsargumas', () {
        final result = checker.check(
          wheel(diameter: 14.2, width: 7.9, et: 45.0),
          octaviaMk2(),
        );
        expect(result.status, FitmentStatus.caution);
        final failCount =
            result.checks.where((c) => c.status == FitmentParameterStatus.fail);
        expect(failCount, isEmpty);
        final warningCount = result.checks
            .where((c) => c.status == FitmentParameterStatus.warning);
        expect(warningCount.length, greaterThanOrEqualTo(2));
      });

      test('kelios įspėjimai + 1 klaida => nesuderinama', () {
        final xtremeTire = wheel(
          diameter: 20.0,
          width: 8.0,
          et: 41.0,
        );
        final result = checker.check(xtremeTire, octaviaMk2());
        expect(result.status, FitmentStatus.incompatible);

        final fails =
            result.checks.where((c) => c.status == FitmentParameterStatus.fail);
        expect(fails.length, greaterThanOrEqualTo(2)); // diameter + ET fail
      });

      test('vienas įspėjimas => atsargumas', () {
        final result = checker.check(wheel(et: 45.0), octaviaMk2());
        expect(result.status, FitmentStatus.caution);
        final warnings = result.checks
            .where((c) => c.status == FitmentParameterStatus.warning);
        expect(warnings.length, 1);
      });

      test('visi parametrai ribose => suderinama', () {
        final result = checker.check(
          wheel(
            diameter: 15.0,
            width: 7.5,
            et: 47.0,
            cb: 57.1,
            pcd: '5x112',
          ),
          octaviaMk2(),
        );
        expect(result.status, FitmentStatus.compatible);
        expect(
          result.checks.every((c) => c.status == FitmentParameterStatus.ok),
          isTrue,
        );
      });
    });

    group('Kraštutiniai atvejai', () {
      test('tiksliai įspėjimo slenkstis (skersmuo)', () {
        // 15.0 - 1.0 = 14.0 turėtų būti įspėjimas
        final result = checker.check(wheel(diameter: 14.0), octaviaMk2());
        final diamCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.diameter,
        );
        expect(diamCheck.status, FitmentParameterStatus.warning);
      });

      test('tiksliai įsp. slenkstis (plotis)', () {
        // 7.5 + 0.5 = 8.0 turėtų būti įspėjimas
        final result = checker.check(wheel(width: 8.0), octaviaMk2());
        final widthCheck = result.checks.firstWhere(
          (c) => c.parameter == FitmentParameter.width,
        );
        expect(widthCheck.status, FitmentParameterStatus.warning);
      });

      test('tiksliai įsp. slenkstis (ET)', () {
        // 54.0 + 5.0 = 59.0 turėtų būti įspėjimas
        final result = checker.check(wheel(et: 59.0), octaviaMk2());
        final etCheck =
            result.checks.firstWhere((c) => c.parameter == FitmentParameter.et);
        expect(etCheck.status, FitmentParameterStatus.warning);
      });
    });

    group('Būsenų agregacija', () {
      test('klaida turi prioritetą prieš įspėjimą => nesuderinama', () {
        final result = checker.check(
          wheel(diameter: 20.0, et: 45.0), // diameter fail, ET warning
          octaviaMk2(),
        );
        expect(result.status, FitmentStatus.incompatible);
      });

      test('bet koks įspėjimas be klaidos => atsargumas', () {
        final result = checker.check(
          wheel(width: 5.6), // single warning
          octaviaMk2(),
        );
        expect(result.status, FitmentStatus.caution);
      });

      test('nėra įsp./klaidų => suderinama', () {
        final result = checker.check(
          wheel(diameter: 17.0, width: 7.0, et: 50.0),
          octaviaMk2(),
        );
        expect(result.status, FitmentStatus.compatible);
      });
    });
  });
}
