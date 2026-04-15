import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../core/state/app_state.dart';
import '../core/theme/app_theme.dart';
import '../features/capture/capture_screen.dart';

class App extends StatelessWidget {
  final AppState appState;

  const App({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider<AppState>.value(
      value: appState,
      child: Consumer<AppState>(
        builder: (context, state, _) {
          return MaterialApp(
            debugShowCheckedModeBanner: false,
            title: 'DudaRims',
            theme: AppTheme.light(),
            darkTheme: AppTheme.dark(),
            themeMode: state.themeMode,
            home: const CaptureScreen(),
          );
        },
      ),
    );
  }
}