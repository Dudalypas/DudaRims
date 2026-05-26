import 'package:flutter/material.dart';

class AppUiTokens {
  static const double primaryActionButtonHeight = 56;
  static const double tertiaryActionButtonHeight = 44;

  static const double fieldLabelBottomSpacing = 8;
  static const double fieldVerticalSpacing = 16;
  static const double buttonVerticalSpacing = 10;
}

extension AppThemeTokens on ColorScheme {
  Color get pageSurface => surface;
  Color get panelSurface => surfaceContainer;
  Color get panelSurfaceHigh => surfaceContainerHigh;
  Color get mutedText => onSurfaceVariant;

  // Kameros overlay spalvos geresniam korteliu kontrastui
  Color get cameraOverlayStrong => shadow.withValues(alpha: 0.42);
  Color get cameraOverlaySoft => shadow.withValues(alpha: 0.34);
  Color get cameraControlFill => surface.withValues(alpha: 0.58);
  Color get cameraControlBorder => outline.withValues(alpha: 0.72);
}