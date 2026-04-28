import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/state/app_state.dart';
import '../../core/theme/theme_tokens.dart';
import '../../models/fitment_result.dart';
import '../../models/wheel_spec.dart';
import '../../services/fitment_checker.dart';
import '../../services/local_fitment_repository.dart';
import '../car/car_selection_screen.dart';

class RimDetailsScreen extends StatefulWidget {
  final File imageFile;
  final String selectedClassName;

  const RimDetailsScreen({
    super.key,
    required this.imageFile,
    required this.selectedClassName,
  });

  @override
  State<RimDetailsScreen> createState() => _RimDetailsScreenState();
}

class _RimDetailsScreenState extends State<RimDetailsScreen> {
  final _repository = LocalFitmentRepository();
  final _checker = FitmentChecker();

  bool _loading = true;
  String? _error;
  List<WheelSpec> _variants = const [];
  WheelSpec? _selectedVariant;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final wheelSpecs = await _repository.loadWheelSpecs();
      final variants = wheelSpecs
          .where(
            (w) =>
                w.wheelClassName.toLowerCase() ==
                widget.selectedClassName.toLowerCase(),
          )
          .toList(growable: false);

      if (!mounted) return;
      setState(() {
        _variants = variants;
        _selectedVariant = variants.isNotEmpty ? variants.first : null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Nepavyko įkelti ratlankio duomenų: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<AppState>();

    return Scaffold(
      appBar: AppBar(title: const Text('Ratlankio detalės')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
          ? Center(child: Text(_error!))
          : _variants.isEmpty
          ? const Center(child: Text('Šio modelio specifikacijos nerastos.'))
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildHeader(),
                  const SizedBox(height: 16),
                  _buildVariantSelector(),
                  const SizedBox(height: 16),
                  if (_selectedVariant != null)
                    _buildSpecCard(_selectedVariant!),
                  const SizedBox(height: 16),
                  _buildCompatibility(appState),
                ],
              ),
            ),
    );
  }

  Widget _buildHeader() {
    final title =
        _selectedVariant?.officialWheelName ?? widget.selectedClassName;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Row(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(14),
              child: Image.file(
                widget.imageFile,
                width: 90,
                height: 90,
                fit: BoxFit.cover,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontWeight: FontWeight.w800,
                      fontSize: 20,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text('Modelio klasė: ${widget.selectedClassName}'),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildVariantSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Variantai',
          style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: _variants
              .map((variant) {
                final selected = identical(variant, _selectedVariant);
                final diameter = variant.diameterIn == null
                    ? '-'
                    : variant.diameterIn!.toStringAsFixed(
                        variant.diameterIn! % 1 == 0 ? 0 : 1,
                      );
                final width = variant.widthJ == null
                    ? '-'
                    : variant.widthJ!.toStringAsFixed(
                        variant.widthJ! % 1 == 0 ? 0 : 1,
                      );
                final et = variant.et == null
                    ? '-'
                    : variant.et!.toStringAsFixed(variant.et! % 1 == 0 ? 0 : 1);
                final label = '$diameter" • ${width}J • ET$et';
                return ChoiceChip(
                  selected: selected,
                  label: Text(label),
                  onSelected: (_) {
                    setState(() {
                      _selectedVariant = variant;
                    });
                  },
                );
              })
              .toList(growable: false),
        ),
      ],
    );
  }

  Widget _buildSpecCard(WheelSpec variant) {
    Widget row(String label, String value) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(
          children: [
            SizedBox(
              width: 120,
              child: Text(
                label,
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
            ),
            Expanded(child: Text(value)),
          ],
        ),
      );
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Specifikacija',
              style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
            ),
            const SizedBox(height: 8),
            row('Diametras', '${variant.diameterIn ?? '-'} col.'),
            row('Plotis', '${variant.widthJ ?? '-'} J'),
            row('ET', '${variant.et ?? '-'}'),
            row('PCD', variant.pcd ?? '-'),
            row('CB', variant.cb?.toString() ?? '-'),
            row(
              'Apdaila',
              variant.finishes.isEmpty ? '-' : variant.finishes.join(', '),
            ),
            row('Modeliai', variant.designedForModels ?? '-'),
          ],
        ),
      ),
    );
  }

  Widget _buildCompatibility(AppState appState) {
    final selectedCar = appState.selectedCar;
    final variant = _selectedVariant;

    if (variant == null) {
      return const SizedBox.shrink();
    }

    if (selectedCar == null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Suderinamumas',
                style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
              ),
              const SizedBox(height: 8),
              Text(
                'Pasirink automobilį, kad galėtume patikrinti suderinamumą.',
                style: TextStyle(
                  color: Theme.of(context).colorScheme.mutedText,
                ),
              ),
              const SizedBox(height: 12),
              FilledButton(
                onPressed: () async {
                  await Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (_) => const CarSelectionScreen(),
                    ),
                  );
                  if (mounted) setState(() {});
                },
                child: const Text('Pasirinkti automobilį'),
              ),
            ],
          ),
        ),
      );
    }

    final result = _checker.check(variant, selectedCar);
    final statusColor = _statusColor(result.status);
    final issues = result.checks
        .where((c) => c.status != FitmentParameterStatus.ok)
        .toList(growable: false);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    'Suderinamumas su ${selectedCar.model ?? 'Skoda'} ${selectedCar.generationLabel}',
                    style: const TextStyle(
                      fontWeight: FontWeight.w700,
                      fontSize: 16,
                    ),
                  ),
                ),
                IconButton(
                  tooltip: 'Keisti automobilį',
                  onPressed: () async {
                    await Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => const CarSelectionScreen(),
                      ),
                    );
                    if (mounted) setState(() {});
                  },
                  icon: const Icon(Icons.edit_outlined),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(14),
                color: statusColor.withValues(alpha: 0.16),
                border: Border.all(color: statusColor),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _statusLabel(result.status),
                    style: TextStyle(
                      fontWeight: FontWeight.w700,
                      color: statusColor,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(_statusDescription(result.status)),
                ],
              ),
            ),
            const SizedBox(height: 10),
            if (issues.isNotEmpty) ...[
              const Text(
                'Paaiškinimai',
                style: TextStyle(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 6),
              ...issues.map(
                (issue) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Text('• ${issue.message}'),
                ),
              ),
            ],
            const SizedBox(height: 8),
            ExpansionTile(
              title: const Text('Techninė detalizacija'),
              children: result.checks
                  .map(
                    (check) => ListTile(
                      dense: true,
                      title: Text(_parameterLabel(check.parameter)),
                      subtitle: Text(
                        'Ratlankis: ${check.rimValue} | Automobilio riba: ${check.expectedValue}',
                      ),
                      trailing: Text(
                        _statusText(check.status),
                        style: TextStyle(
                          color: _statusColorByCheck(check.status),
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                  )
                  .toList(growable: false),
            ),
          ],
        ),
      ),
    );
  }

  String _parameterLabel(FitmentParameter parameter) {
    switch (parameter) {
      case FitmentParameter.diameter:
        return 'Skersmuo';
      case FitmentParameter.width:
        return 'Plotis';
      case FitmentParameter.et:
        return 'ET';
      case FitmentParameter.pcd:
        return 'PCD';
      case FitmentParameter.cb:
        return 'CB';
    }
  }

  String _statusText(FitmentParameterStatus status) {
    switch (status) {
      case FitmentParameterStatus.ok:
        return 'OK';
      case FitmentParameterStatus.warning:
        return 'Pastaba';
      case FitmentParameterStatus.fail:
        return 'Kritinis';
    }
  }

  Color _statusColorByCheck(FitmentParameterStatus status) {
    switch (status) {
      case FitmentParameterStatus.ok:
        return Colors.green;
      case FitmentParameterStatus.warning:
        return Colors.orange;
      case FitmentParameterStatus.fail:
        return Colors.red;
    }
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

  String _statusLabel(FitmentStatus status) {
    switch (status) {
      case FitmentStatus.compatible:
        return 'Suderinama';
      case FitmentStatus.caution:
        return 'Suderinama su pastabomis';
      case FitmentStatus.incompatible:
        return 'Nesuderinama';
    }
  }

  String _statusDescription(FitmentStatus status) {
    switch (status) {
      case FitmentStatus.compatible:
        return 'Visi pagrindiniai parametrai atitinka pasirinktą automobilį.';
      case FitmentStatus.caution:
        return 'Kai kurie parametrai yra ribiniai arba reikalauja papildomos patikros.';
      case FitmentStatus.incompatible:
        return 'Vienas ar keli kritiniai parametrai nesutampa.';
    }
  }
}
