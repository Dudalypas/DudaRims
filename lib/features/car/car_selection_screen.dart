import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/state/app_state.dart';
import '../../core/theme/theme_tokens.dart';
import '../../models/vehicle_fitment.dart';
import '../../services/local_fitment_repository.dart';

class CarSelectionScreen extends StatefulWidget {
  const CarSelectionScreen({super.key});

  @override
  State<CarSelectionScreen> createState() => _CarSelectionScreenState();
}

class _CarSelectionScreenState extends State<CarSelectionScreen> {
  final _repo = LocalFitmentRepository();

  bool _loading = true;
  String? _error;
  List<VehicleFitment> _fitments = const [];

  String? _selectedModel;
  VehicleFitment? _selectedGeneration;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final appState = context.read<AppState>();
    final selectedCar = appState.selectedCar;

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final fitments = await _repo.loadVehicleFitments();
      final skoda = fitments
          .where((f) => (f.brand ?? '').toLowerCase() == 'skoda')
          .toList(growable: false);

      final models = _models(skoda);
      final initialModel =
          selectedCar?.model ?? (models.isNotEmpty ? models.first : null);
      final generations = _generations(skoda, initialModel);

      VehicleFitment? initialGeneration;
      if (selectedCar != null) {
        initialGeneration = generations
            .where((g) => g.generationLabel == selectedCar.generationLabel)
            .cast<VehicleFitment?>()
            .firstOrNull;
      }
      initialGeneration ??= generations.isNotEmpty ? generations.first : null;

      if (!mounted) return;
      setState(() {
        _fitments = skoda;
        _selectedModel = initialModel;
        _selectedGeneration = initialGeneration;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Nepavyko įkelti automobilių: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  List<String> _models(List<VehicleFitment> fitments) {
    final out = <String>[];
    for (final fit in fitments) {
      final model = (fit.model ?? '').trim();
      if (model.isEmpty) continue;
      if (!out.any((m) => m.toLowerCase() == model.toLowerCase())) {
        out.add(model);
      }
    }
    out.sort((a, b) => a.toLowerCase().compareTo(b.toLowerCase()));
    return out;
  }

  List<VehicleFitment> _generations(
    List<VehicleFitment> fitments,
    String? model,
  ) {
    if (model == null) return const [];
    return fitments
        .where((f) => (f.model ?? '').toLowerCase() == model.toLowerCase())
        .toList(growable: false);
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(title: const Text('Pasirinkti automobilį')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? Center(child: Text(_error!))
          : Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: colors.panelSurfaceHigh,
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Text(
                      'Skoda modelis ir karta bus išsaugoti bei naudojami suderinamumo tikrinime.',
                      style: TextStyle(color: colors.mutedText),
                    ),
                  ),
                  const SizedBox(height: 16),
                  DropdownButtonFormField<String>(
                    key: ValueKey('model-${_selectedModel ?? ''}'),
                    initialValue: _selectedModel,
                    items: _models(_fitments)
                        .map(
                          (m) => DropdownMenuItem<String>(
                            value: m,
                            child: Text(m),
                          ),
                        )
                        .toList(growable: false),
                    onChanged: (value) {
                      final generations = _generations(_fitments, value);
                      setState(() {
                        _selectedModel = value;
                        _selectedGeneration = generations.isNotEmpty
                            ? generations.first
                            : null;
                      });
                    },
                    decoration: const InputDecoration(labelText: 'Modelis'),
                  ),
                  const SizedBox(height: 12),
                  DropdownButtonFormField<VehicleFitment>(
                    key: ValueKey(
                      'gen-${_selectedGeneration?.generationLabel ?? ''}',
                    ),
                    initialValue: _selectedGeneration,
                    items: _generations(_fitments, _selectedModel)
                        .map(
                          (g) => DropdownMenuItem<VehicleFitment>(
                            value: g,
                            child: Text(g.generationLabel),
                          ),
                        )
                        .toList(growable: false),
                    onChanged: (value) {
                      setState(() {
                        _selectedGeneration = value;
                      });
                    },
                    decoration: const InputDecoration(
                      labelText: 'Karta / metai',
                    ),
                  ),
                  const Spacer(),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: _selectedGeneration == null
                          ? null
                          : () async {
                              await context.read<AppState>().setSelectedCar(
                                _selectedGeneration,
                              );
                              if (!context.mounted) return;
                              Navigator.of(context).pop();
                            },
                      child: const Text('Išsaugoti automobilį'),
                    ),
                  ),
                  const SizedBox(height: 8),
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton(
                      onPressed: () async {
                        await context.read<AppState>().setSelectedCar(null);
                        if (!context.mounted) return;
                        Navigator.of(context).pop();
                      },
                      child: const Text('Pašalinti pasirinktą automobilį'),
                    ),
                  ),
                ],
              ),
            ),
    );
  }
}

extension<T> on Iterable<T> {
  T? get firstOrNull => isEmpty ? null : first;
}
