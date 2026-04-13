enum FitmentStatus { compatible, caution, incompatible }

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
  final List<FitmentReason> reasons;

  const FitmentResult({
    required this.status,
    required this.reasons,
  });
}
