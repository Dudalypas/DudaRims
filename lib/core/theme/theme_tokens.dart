import 'package:flutter/material.dart';

extension AppThemeTokens on ColorScheme {
  Color get pageSurface => surface;
  Color get panelSurface => surfaceContainer;
  Color get panelSurfaceHigh => surfaceContainerHigh;
  Color get mutedText => onSurfaceVariant;

  // Camera screen keeps an intentional dark overlay for contrast with preview.
  Color get cameraOverlayStrong => shadow.withValues(alpha: 0.42);
  Color get cameraOverlaySoft => shadow.withValues(alpha: 0.34);
  Color get cameraControlFill => surface.withValues(alpha: 0.58);
  Color get cameraControlBorder => outline.withValues(alpha: 0.72);
}
