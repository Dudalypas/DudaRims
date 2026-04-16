class AppConstants {
  static const double recognitionStrongThreshold = 0.82;
  static const double recognitionLowConfidenceThreshold = 0.15;

  static const String recognitionFailureMessage =
      'Pabandyk įkelti aiškesnę nuotrauką, kurioje ratlankis matomas iš priekio ir užima didžiąją kadro dalį.';

  // Keep mock enabled by default for predictable demo flow.
  // Switch to true when production model assets are validated end-to-end.
  static const bool enableRealMlInference = true;

  // Fitment grading thresholds.
  static const double fitmentDiameterWarningDeviationIn = 1.0;
  static const double fitmentWidthWarningDeviationJ = 0.5;
  static const double fitmentEtWarningDeviationMm = 5.0;
}
