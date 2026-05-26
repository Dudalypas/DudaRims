import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/state/app_state.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  static const List<(String, String)> _faqItems = [
    (
      'Kaip veikia ratlankio atpažinimas?',
      'Programėlė palygina tavo nuotrauką su lokalia ratlankių duomenų baze ir pateikia labiausiai panašų modelį arba kelis galimus variantus.',
    ),
    (
      'Ką reiškia identifikavimo atitikimas?',
      'Tai yra tik modelio identifikavimo tikimybė. Ji neparodo, ar ratlankis tinka tavo automobiliui.',
    ),
    (
      'Kodėl kartais pateikiami keli galimi variantai?',
      'Kai nuotrauka nepakankamai aiški arba keli modeliai vizualiai panašūs, programa siūlo kelis artimiausius variantus.',
    ),
    (
      'Kaip tikrinamas suderinamumas?',
      'Suderinamumas skaičiuojamas atskirai pagal techninius parametrus: PCD, ET, CB, plotį ir skersmenį, palyginant su pasirinktu automobiliu.',
    ),
    (
      'Ar spalvinis / apdailos variantas keičia suderinamumą?',
      'Ne. Spalva ar apdaila yra kosmetika. Suderinamumą lemia tik techniniai matmenys ir tvirtinimo parametrai.',
    ),
  ];

  static const List<(String, String)> _termItems = [
    (
      'PCD',
      'Varžtų išdėstymas. Turi sutapti tiksliai, kitaip ratlankis netiks.',
    ),
    (
      'ET',
      'Išnešimas. Parodo, kiek ratlankis pasislinkęs į vidų ar išorę. Per didelis nukrypimas gali sukelti trynimąsi.',
    ),
    (
      'CB',
      'Centrinės skylės skersmuo. Jei per maža - neužsidės, jei didesnė - gali reikėti centravimo žiedų.',
    ),
    (
      'Ratlankio plotis',
      'Ratlankio plotis (J) turi būti tinkamas padangai ir automobilio leistinoms riboms.',
    ),
    (
      'Ratlankio skersmuo',
      'Ratlankio skersmuo coliais. Turi atitikti padangą ir gamintojo rekomenduojamą diapazoną.',
    ),
  ];

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
              subtitle: const Text(
                'Perjungti tarp šviesios ir tamsios išvaizdos.',
              ),
              value: isDark,
              onChanged: (value) {
                appState.setThemeMode(value ? ThemeMode.dark : ThemeMode.light);
              },
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: ExpansionTile(
              title: const Text('DUK'),
              subtitle: const Text('Trumpi atsakymai pradedantiesiems.'),
              children: _faqItems
                  .map(
                    (item) =>
                        ListTile(title: Text(item.$1), subtitle: Text(item.$2)),
                  )
                  .toList(growable: false),
            ),
          ),
          const SizedBox(height: 8),
          Card(
            child: ExpansionTile(
              title: const Text('PCD / ET / CB ir pagrindiniai terminai'),
              subtitle: const Text('Trumpi praktiniai paaiškinimai.'),
              children: _termItems
                  .map(
                    (item) =>
                        ListTile(title: Text(item.$1), subtitle: Text(item.$2)),
                  )
                  .toList(growable: false),
            ),
          ),
        ],
      ),
    );
  }
}
