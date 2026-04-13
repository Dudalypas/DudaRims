import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../models/vehicle_fitment.dart';
import '../../models/wheel_spec.dart';
import '../../services/local_fitment_repository.dart';

class AppState extends ChangeNotifier {
  static const String themeModeKey = 'theme_mode';
  static const String selectedCarKey = 'selected_car';

  static const double confidenceThreshold = 0.72;

  final LocalFitmentRepository repository;
  final SharedPreferences prefs;

  ThemeMode _themeMode = ThemeMode.dark;
  VehicleFitment? _selectedCar;
  bool _isLoadingData = true;
  String? _dataError;
  List<WheelSpec> _wheelSpecs = const [];
  List<VehicleFitment> _vehicleFitments = const [];

  AppState({required this.repository, required this.prefs});

  ThemeMode get themeMode => _themeMode;
  VehicleFitment? get selectedCar => _selectedCar;
  bool get isLoadingData => _isLoadingData;
  String? get dataError => _dataError;
  List<WheelSpec> get wheelSpecs => _wheelSpecs;
  List<VehicleFitment> get vehicleFitments => _vehicleFitments;

  static Future<AppState> create() async {
    final prefs = await SharedPreferences.getInstance();
    final state = AppState(repository: LocalFitmentRepository(), prefs: prefs);
    await state._initialize();
    return state;
  }

  Future<void> _initialize() async {
    final modeRaw = prefs.getString(themeModeKey);
    if (modeRaw == 'light') {
      _themeMode = ThemeMode.light;
    } else if (modeRaw == 'dark') {
      _themeMode = ThemeMode.dark;
    } else {
      _themeMode = ThemeMode.dark;
    }

    await loadCatalogData();
    _restoreSelectedCar();
  }

  Future<void> loadCatalogData() async {
    _isLoadingData = true;
    _dataError = null;
    notifyListeners();

    try {
      _wheelSpecs = await repository.loadWheelSpecs();
      _vehicleFitments = await repository.loadVehicleFitments();
    } catch (e) {
      _dataError = 'Nepavyko užkrauti duomenų: $e';
    } finally {
      _isLoadingData = false;
      notifyListeners();
    }
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    _themeMode = mode;
    await prefs.setString(themeModeKey, mode == ThemeMode.light ? 'light' : 'dark');
    notifyListeners();
  }

  Future<void> setSelectedCar(VehicleFitment? fitment) async {
    _selectedCar = fitment;
    if (fitment == null) {
      await prefs.remove(selectedCarKey);
    } else {
      final payload = {
        'brand': fitment.brand,
        'model': fitment.model,
        'generation': fitment.generation,
        'year_from': fitment.yearFrom,
        'year_to': fitment.yearTo,
      };
      await prefs.setString(selectedCarKey, jsonEncode(payload));
    }
    notifyListeners();
  }

  void _restoreSelectedCar() {
    final raw = prefs.getString(selectedCarKey);
    if (raw == null || raw.trim().isEmpty || _vehicleFitments.isEmpty) {
      return;
    }

    try {
      final decoded = jsonDecode(raw);
      if (decoded is! Map<String, dynamic>) return;

      final brand = (decoded['brand'] ?? '').toString().toLowerCase();
      final model = (decoded['model'] ?? '').toString().toLowerCase();
      final generation = (decoded['generation'] ?? '').toString().toLowerCase();
      final yearFrom = decoded['year_from'];
      final yearTo = decoded['year_to'];

      for (final v in _vehicleFitments) {
        final sameBrand = (v.brand ?? '').toLowerCase() == brand;
        final sameModel = (v.model ?? '').toLowerCase() == model;
        final sameGeneration = (v.generation ?? '').toLowerCase() == generation;
        final sameYears = v.yearFrom == yearFrom && v.yearTo == yearTo;
        if (sameBrand && sameModel && sameGeneration && sameYears) {
          _selectedCar = v;
          break;
        }
      }
    } catch (_) {
      // Ignore invalid persisted payload.
    }
  }
}
