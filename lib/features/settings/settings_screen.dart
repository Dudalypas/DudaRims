import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../app/state/app_state.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<AppState>();
    final isDark = appState.themeMode == ThemeMode.dark;

    return Scaffold(
      appBar: AppBar(title: const Text('Nustatymai')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: SwitchListTile.adaptive(
              value: isDark,
              title: const Text('Tamsi tema'),
              subtitle: const Text('Numatytoji premium išvaizda'),
              onChanged: (value) => appState.setThemeMode(value ? ThemeMode.dark : ThemeMode.light),
            ),
          ),
          const SizedBox(height: 12),
          const Card(
            child: ListTile(
              title: Text('PCD / ET / CB pagrindai'),
              subtitle: Text('Greitai: PCD ir varžtai turi sutapti, ET ir plotis turi likti OEM ribose.'),
            ),
          ),
          const SizedBox(height: 12),
          const Card(
            child: ListTile(
              title: Text('DUK (vieta ateičiai)'),
              subtitle: Text('Čia galite pridėti trumpus fitment paaiškinimus ir patarimus.'),
            ),
          ),
        ],
      ),
    );
  }
}
