import 'package:flutter/material.dart';

import '../theme/theme_tokens.dart';

class ActionButtonStyles {
  static Widget primaryButton({
    required BuildContext context,
    required VoidCallback? onPressed,
    required String label,
    Widget? icon,
  }) {
    return SizedBox(
      width: double.infinity,
      height: AppUiTokens.primaryActionButtonHeight,
      child: icon != null
          ? FilledButton.icon(
              onPressed: onPressed,
              icon: icon,
              label: Text(label),
            )
          : FilledButton(
              onPressed: onPressed,
              child: Text(label),
            ),
    );
  }

  static Widget secondaryButton({
    required BuildContext context,
    required VoidCallback? onPressed,
    required String label,
    Widget? icon,
  }) {
    return SizedBox(
      width: double.infinity,
      height: AppUiTokens.primaryActionButtonHeight,
      child: icon != null
          ? OutlinedButton.icon(
              onPressed: onPressed,
              icon: icon,
              label: Text(label),
            )
          : OutlinedButton(
              onPressed: onPressed,
              child: Text(label),
            ),
    );
  }

  static Widget tertiaryDestructiveButton({
    required BuildContext context,
    required VoidCallback? onPressed,
    required String label,
  }) {
    final colors = Theme.of(context).colorScheme;

    return SizedBox(
      width: double.infinity,
      height: 44,
      child: TextButton(
        onPressed: onPressed,
        style: TextButton.styleFrom(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 16),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: colors.error,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
    );
  }

  static const double buttonStackSpacing = AppUiTokens.buttonVerticalSpacing;
}