import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../models/vehicle_fitment.dart';

class AppState extends ChangeNotifier {
  static const _themeKey = 'app.theme_mode';
  static const _carKey = 'app.selected_car';

  ThemeMode _themeMode = ThemeMode.dark;
  VehicleFitment? _selectedCar;

  ThemeMode get themeMode => _themeMode;
  VehicleFitment? get selectedCar => _selectedCar;

  Future<void> initialize() async {
    final prefs = await SharedPreferences.getInstance();

    // Atkuriami lokaliai issaugoti naudotojo pasirinkimai
    final savedTheme = prefs.getString(_themeKey);
    if (savedTheme == 'light') {
      _themeMode = ThemeMode.light;
    } else if (savedTheme == 'dark') {
      _themeMode = ThemeMode.dark;
    }

    final savedCarJson = prefs.getString(_carKey);
    if (savedCarJson != null && savedCarJson.isNotEmpty) {
      try {
        final decoded = jsonDecode(savedCarJson);
        if (decoded is Map<String, dynamic>) {
          _selectedCar = VehicleFitment.fromJson(decoded);
        } else if (decoded is Map) {
          _selectedCar = VehicleFitment.fromJson(decoded.cast<String, dynamic>());
        }
      } catch (_) {
        // Jei irasas sugadintas arba seno formato, pasirinkima atmetam
        _selectedCar = null;
      }
    }
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    _themeMode = mode;
    notifyListeners();

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_themeKey, mode == ThemeMode.light ? 'light' : 'dark');
  }

  Future<void> setSelectedCar(VehicleFitment? car) async {
    _selectedCar = car;
    notifyListeners();

    final prefs = await SharedPreferences.getInstance();
    if (car == null) {
      await prefs.remove(_carKey);
    } else {
      await prefs.setString(_carKey, jsonEncode(car.toJson()));
    }
  }
}