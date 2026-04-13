import 'package:flutter/material.dart';

import '../../models/vehicle_fitment.dart';

class CarPickerSheet extends StatefulWidget {
  final List<VehicleFitment> fitments;
  final VehicleFitment? initialSelection;

  const CarPickerSheet({
    super.key,
    required this.fitments,
    this.initialSelection,
  });

  @override
  State<CarPickerSheet> createState() => _CarPickerSheetState();
}

class _CarPickerSheetState extends State<CarPickerSheet> {
  String _brand = 'Skoda';
  String? _model;
  VehicleFitment? _generation;

  @override
  void initState() {
    super.initState();
    if (widget.initialSelection != null) {
      _brand = widget.initialSelection!.brand ?? 'Skoda';
      _model = widget.initialSelection!.model;
      _generation = widget.initialSelection;
    } else {
      final models = _modelsForBrand(_brand);
      _model = models.isNotEmpty ? models.first : null;
      final gens = _generations(_brand, _model);
      _generation = gens.isNotEmpty ? gens.first : null;
    }
  }

  List<String> _modelsForBrand(String brand) {
    final models = <String>[];
    for (final v in widget.fitments) {
      if ((v.brand ?? '').toLowerCase() != brand.toLowerCase()) continue;
      final model = (v.model ?? '').trim();
      if (model.isEmpty) continue;
      if (!models.any((m) => m.toLowerCase() == model.toLowerCase())) {
        models.add(model);
      }
    }
    models.sort((a, b) => a.toLowerCase().compareTo(b.toLowerCase()));
    return models;
  }

  List<VehicleFitment> _generations(String brand, String? model) {
    return widget.fitments
        .where((v) =>
            (v.brand ?? '').toLowerCase() == brand.toLowerCase() &&
            (model == null || (v.model ?? '').toLowerCase() == model.toLowerCase()))
        .toList();
  }

  @override
  Widget build(BuildContext context) {
    final models = _modelsForBrand(_brand);
    final generations = _generations(_brand, _model);

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 44,
              height: 4,
              margin: const EdgeInsets.only(bottom: 12),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.outlineVariant,
                borderRadius: BorderRadius.circular(999),
              ),
            ),
            const Text('Pasirink automobilį', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700)),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              initialValue: _brand,
              key: ValueKey('brand-$_brand'),
              items: const [DropdownMenuItem(value: 'Skoda', child: Text('Skoda'))],
              onChanged: (v) {
                if (v == null) return;
                final modelList = _modelsForBrand(v);
                final nextModel = modelList.isNotEmpty ? modelList.first : null;
                final nextGens = _generations(v, nextModel);
                setState(() {
                  _brand = v;
                  _model = nextModel;
                  _generation = nextGens.isNotEmpty ? nextGens.first : null;
                });
              },
              decoration: const InputDecoration(labelText: 'Markė'),
            ),
            const SizedBox(height: 10),
            DropdownButtonFormField<String>(
              key: ValueKey('model-${_model ?? ''}'),
              initialValue: models.contains(_model) ? _model : null,
              items: models.map((m) => DropdownMenuItem(value: m, child: Text(m))).toList(growable: false),
              onChanged: (v) {
                final nextGens = _generations(_brand, v);
                setState(() {
                  _model = v;
                  _generation = nextGens.isNotEmpty ? nextGens.first : null;
                });
              },
              decoration: const InputDecoration(labelText: 'Modelis'),
            ),
            const SizedBox(height: 10),
            DropdownButtonFormField<VehicleFitment>(
              key: ValueKey('generation-${_generation?.generationLabel ?? ''}'),
              initialValue: generations.contains(_generation) ? _generation : null,
              items: generations
                  .map((g) => DropdownMenuItem<VehicleFitment>(value: g, child: Text(g.generationLabel)))
                  .toList(growable: false),
              onChanged: (v) => setState(() => _generation = v),
              decoration: const InputDecoration(labelText: 'Karta / metai'),
            ),
            const SizedBox(height: 14),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: _generation == null ? null : () => Navigator.of(context).pop(_generation),
                child: const Text('Išsaugoti pasirinkimą'),
              ),
            ),
            const SizedBox(height: 6),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton(
                onPressed: () => Navigator.of(context).pop(null),
                child: const Text('Pašalinti automobilį'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
