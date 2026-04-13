import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../app/state/app_state.dart';
import '../../models/fitment_result.dart';
import '../../models/rim_model.dart';
import '../../models/rim_variant.dart';
import '../../services/fitment_checker.dart';
import '../car/car_picker_sheet.dart';

class RimDetailScreen extends StatefulWidget {
  final File previewImage;
  final RimModel rimModel;

  const RimDetailScreen({
    super.key,
    required this.previewImage,
    required this.rimModel,
  });

  @override
  State<RimDetailScreen> createState() => _RimDetailScreenState();
}

class _RimDetailScreenState extends State<RimDetailScreen> {
  final _checker = FitmentChecker();
  late RimVariant _selectedVariant;

  @override
  void initState() {
    super.initState();
    _selectedVariant = widget.rimModel.variants.first;
  }

  Color _statusColor(FitmentStatus status) {
    switch (status) {
      case FitmentStatus.compatible:
        return Colors.green;
      case FitmentStatus.caution:
        return Colors.orange;
      case FitmentStatus.incompatible:
        return Colors.red;
    }
  }

  String _statusText(FitmentStatus status) {
    switch (status) {
      case FitmentStatus.compatible:
        return 'Tinka';
      case FitmentStatus.caution:
        return 'Tinka su pastabomis';
      case FitmentStatus.incompatible:
        return 'Netinka';
    }
  }

  String _statusDesc(FitmentStatus status) {
    switch (status) {
      case FitmentStatus.compatible:
        return 'Visi pagrindiniai parametrai atitinka pasirinktą automobilį.';
      case FitmentStatus.caution:
        return 'Dalis parametrų yra ribiniai arba trūksta duomenų.';
      case FitmentStatus.incompatible:
        return 'Yra kritinių parametrų neatitikimų.';
    }
  }

  Future<void> _changeCar(BuildContext context) async {
    final appState = context.read<AppState>();
    final selected = await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (_) => CarPickerSheet(
        fitments: appState.vehicleFitments,
        initialSelection: appState.selectedCar,
      ),
    );

    if (!context.mounted) return;
    await appState.setSelectedCar(selected);
  }

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<AppState>();
    final selectedCar = appState.selectedCar;
    final fitment = selectedCar == null ? null : _checker.check(_selectedVariant.source, selectedCar);

    return Scaffold(
      appBar: AppBar(title: const Text('Ratlankio detalės')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(14),
                    child: Image.file(widget.previewImage, width: 96, height: 96, fit: BoxFit.cover),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(widget.rimModel.displayName, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w700)),
                        const SizedBox(height: 4),
                        Text('Klasė: ${widget.rimModel.className}'),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Variantai', style: TextStyle(fontWeight: FontWeight.w700)),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: widget.rimModel.variants
                        .map(
                          (v) => ChoiceChip(
                            label: Text(v.variantLabel),
                            selected: identical(v, _selectedVariant),
                            onSelected: (_) => setState(() => _selectedVariant = v),
                          ),
                        )
                        .toList(growable: false),
                  ),
                  const SizedBox(height: 12),
                  _SpecGrid(source: _selectedVariant),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Expanded(
                        child: Text('Automobilis', style: TextStyle(fontWeight: FontWeight.w700)),
                      ),
                      TextButton(onPressed: () => _changeCar(context), child: const Text('Keisti')),
                    ],
                  ),
                  if (selectedCar == null)
                    const Text('Pasirink automobilį, kad matytum suderinamumą.')
                  else ...[
                    Text('${selectedCar.brand} ${selectedCar.model}'),
                    Text(selectedCar.generationLabel),
                  ],
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          if (selectedCar != null && fitment != null)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: _statusColor(fitment.status).withValues(alpha: 0.14),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: _statusColor(fitment.status)),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _statusText(fitment.status),
                            style: TextStyle(
                              fontWeight: FontWeight.w700,
                              color: _statusColor(fitment.status),
                              fontSize: 18,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(_statusDesc(fitment.status)),
                        ],
                      ),
                    ),
                    const SizedBox(height: 8),
                    ExpansionTile(
                      tilePadding: EdgeInsets.zero,
                      childrenPadding: const EdgeInsets.only(bottom: 8),
                      title: const Text('Techninis išskaidymas'),
                      children: fitment.reasons.isEmpty
                          ? const [Padding(padding: EdgeInsets.all(8), child: Text('Kritinių neatitikimų nerasta.'))]
                          : fitment.reasons
                              .map(
                                (r) => ListTile(
                                  dense: true,
                                  leading: Icon(
                                    r.isHardFail ? Icons.block : Icons.info_outline,
                                    color: r.isHardFail ? Colors.red : Colors.orange,
                                  ),
                                  title: Text(r.message),
                                ),
                              )
                              .toList(growable: false),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _SpecGrid extends StatelessWidget {
  final RimVariant source;

  const _SpecGrid({required this.source});

  @override
  Widget build(BuildContext context) {
    final items = <MapEntry<String, String>>[
      MapEntry('Diametras', source.source.diameterIn?.toString() ?? '-'),
      MapEntry('Plotis', source.source.widthJ?.toString() ?? '-'),
      MapEntry('ET', source.source.et?.toString() ?? '-'),
      MapEntry('PCD', source.source.pcd ?? '-'),
      MapEntry('CB', source.source.cb?.toString() ?? '-'),
      MapEntry('Varžtai', source.source.boltCount?.toString() ?? '-'),
    ];

    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: items.length,
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        mainAxisExtent: 64,
        crossAxisSpacing: 8,
        mainAxisSpacing: 8,
      ),
      itemBuilder: (context, i) {
        final item = items[i];
        return Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(item.key, style: const TextStyle(fontSize: 12)),
              const SizedBox(height: 2),
              Text(item.value, style: const TextStyle(fontWeight: FontWeight.w700)),
            ],
          ),
        );
      },
    );
  }
}
