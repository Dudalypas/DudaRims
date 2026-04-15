import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/state/app_state.dart';

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
            child: SwitchListTile(
              title: const Text('Tamsi tema'),
              subtitle: const Text('Perjungti tarp šviesios ir tamsios išvaizdos.'),
              value: isDark,
              onChanged: (value) {
                appState.setThemeMode(value ? ThemeMode.dark : ThemeMode.light);
              },
            ),
          ),
          const SizedBox(height: 12),
          const Card(
            child: ListTile(
              title: Text('DUK (netrukus)'),
              subtitle: Text('Trumpi atsakymai apie suderinamumą ir ratlankių žymėjimus.'),
            ),
          ),
          const SizedBox(height: 8),
          const Card(
            child: ListTile(
              title: Text('Apie PCD / ET / CB (netrukus)'),
              subtitle: Text('Paaiškinimai pradedantiesiems ir praktiniai pavyzdžiai.'),
            ),
          ),
        ],
      ),
    );
  }
}
