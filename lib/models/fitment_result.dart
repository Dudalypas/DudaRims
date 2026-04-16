enum FitmentStatus { compatible, caution, incompatible }

enum FitmentParameter { diameter, width, et, pcd, cb }

enum FitmentParameterStatus { ok, warning, fail }

class FitmentParameterCheck {
  final FitmentParameter parameter;
  final String rimValue;
  final String expectedValue;
  final FitmentParameterStatus status;
  final String message;
  final bool isCritical;

  const FitmentParameterCheck({
    required this.parameter,
    required this.rimValue,
    required this.expectedValue,
    required this.status,
    required this.message,
    this.isCritical = false,
  });
}

class FitmentReason {
  final String code;
  final String message;
  final bool isHardFail;

  const FitmentReason({
    required this.code,
    required this.message,
    this.isHardFail = false,
  });
}

class FitmentResult {
  final FitmentStatus status;
  final List<FitmentParameterCheck> checks;

  const FitmentResult({required this.status, required this.checks});

  List<FitmentReason> get reasons {
    return checks
        .where((c) => c.status != FitmentParameterStatus.ok)
        .map(
          (c) => FitmentReason(
            code: '${c.parameter.name}_${c.status.name}',
            message: c.message,
            isHardFail: c.status == FitmentParameterStatus.fail,
          ),
        )
        .toList(growable: false);
  }
}
