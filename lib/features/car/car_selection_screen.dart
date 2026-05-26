import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../core/components/action_button_styles.dart';
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

      String? initialModel;
      VehicleFitment? initialGeneration;

      if (selectedCar != null) {
        final selectedModel = (selectedCar.model ?? '').trim();
        if (selectedModel.isNotEmpty) {
          initialModel = selectedModel;
          final generations = _generations(skoda, initialModel);
          initialGeneration = generations
              .where((g) => g.generationLabel == selectedCar.generationLabel)
              .cast<VehicleFitment?>()
              .firstOrNull;
        }
      }

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

  Future<String?> _pickModel(BuildContext context) async {
    final models = _models(_fitments);
    if (models.isEmpty) return null;

    final colors = Theme.of(context).colorScheme;
    return showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      builder: (sheetContext) {
        var query = '';

        return StatefulBuilder(
          builder: (context, setSheetState) {
            final filteredModels = models
                .where(
                  (model) => model.toLowerCase().contains(query.toLowerCase()),
                )
                .toList(growable: false);

            return Padding(
              padding: EdgeInsets.only(
                bottom: MediaQuery.of(context).viewInsets.bottom,
              ),
              child: SizedBox(
                height: MediaQuery.sizeOf(context).height * 0.78,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 12),
                    Center(
                      child: Container(
                        width: 40,
                        height: 4,
                        decoration: BoxDecoration(
                          color: colors.outlineVariant,
                          borderRadius: BorderRadius.circular(999),
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      child: TextField(
                        autofocus: true,
                        decoration: const InputDecoration(
                          hintText: 'Ieškoti modelio',
                          prefixIcon: Icon(Icons.search),
                        ),
                        onChanged: (value) {
                          setSheetState(() {
                            query = value;
                          });
                        },
                      ),
                    ),
                    const SizedBox(height: 12),
                    Expanded(
                      child: filteredModels.isEmpty
                          ? Center(
                              child: Text(
                                'Modelių nerasta',
                                style: TextStyle(color: colors.mutedText),
                              ),
                            )
                          : ListView.separated(
                              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                              itemCount: filteredModels.length,
                              separatorBuilder: (_, index) => const SizedBox(
                                height: 8,
                              ),
                              itemBuilder: (_, index) {
                                final model = filteredModels[index];
                                final isSelected = model == _selectedModel;

                                return Material(
                                  color: isSelected
                                      ? colors.primaryContainer
                                      : colors.surfaceContainerHigh,
                                  borderRadius: BorderRadius.circular(14),
                                  child: InkWell(
                                    borderRadius: BorderRadius.circular(14),
                                    onTap: () {
                                      Navigator.of(sheetContext).pop(model);
                                    },
                                    child: Padding(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 16,
                                        vertical: 14,
                                      ),
                                      child: Row(
                                        children: [
                                          Expanded(child: Text(model)),
                                          if (isSelected)
                                            Icon(
                                              Icons.check_rounded,
                                              color: colors.primary,
                                            ),
                                        ],
                                      ),
                                    ),
                                  ),
                                );
                              },
                            ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  Widget _fieldLabel(BuildContext context, String label) {
    return Text(label, style: Theme.of(context).textTheme.labelLarge);
  }

  Widget _fieldShell({
    required BuildContext context,
    required Widget child,
    required VoidCallback? onTap,
    required bool showChevron,
  }) {
    final colors = Theme.of(context).colorScheme;

    return Material(
      color: colors.surfaceContainerHighest.withValues(alpha: 0.55),
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
          child: Row(
            children: [
              Expanded(child: child),
              if (showChevron)
                Icon(
                  Icons.expand_more_rounded,
                  color: colors.onSurfaceVariant,
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildGenerationPicker(
    BuildContext context,
    List<VehicleFitment> generations,
  ) {
    final colors = Theme.of(context).colorScheme;

    return Padding(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
      ),
      child: SizedBox(
        height: MediaQuery.sizeOf(context).height * 0.6,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 12),
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: colors.outlineVariant,
                  borderRadius: BorderRadius.circular(999),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Text(
                'Pasirinkite kartą',
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ),
            const SizedBox(height: 12),
            Expanded(
              child: ListView.separated(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                itemCount: generations.length,
                separatorBuilder: (_, index) => const SizedBox(height: 8),
                itemBuilder: (_, index) {
                  final gen = generations[index];
                  final isSelected = gen == _selectedGeneration;

                  return Material(
                    color: isSelected
                        ? colors.primaryContainer
                        : colors.surfaceContainerHigh,
                    borderRadius: BorderRadius.circular(14),
                    child: InkWell(
                      borderRadius: BorderRadius.circular(14),
                      onTap: () {
                        Navigator.of(context).pop(gen);
                      },
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 14,
                        ),
                        child: Row(
                          children: [
                            Expanded(child: Text(gen.generationLabel)),
                            if (isSelected)
                              Icon(
                                Icons.check_rounded,
                                color: colors.primary,
                              ),
                          ],
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
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
                      'Pasirinktas automobilis bus naudojamas suderinamumo tikrinimui.',
                      style: TextStyle(color: colors.mutedText),
                    ),
                  ),
                  const SizedBox(height: 16),
                  _fieldLabel(context, 'Markė'),
                  const SizedBox(height: AppUiTokens.fieldLabelBottomSpacing),
                  _fieldShell(
                    context: context,
                    onTap: null,
                    showChevron: false,
                    child: Text(
                      'Škoda',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: colors.onSurface,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  const SizedBox(height: AppUiTokens.fieldVerticalSpacing),
                  _fieldLabel(context, 'Modelis'),
                  const SizedBox(height: AppUiTokens.fieldLabelBottomSpacing),
                  _fieldShell(
                    context: context,
                    onTap: () async {
                      final selectedModel = await _pickModel(context);
                      if (selectedModel == null || !mounted) return;
                      setState(() {
                        _selectedModel = selectedModel;
                        _selectedGeneration = null;
                      });
                    },
                    showChevron: true,
                    child: Text(
                      _selectedModel ?? 'Pasirinkite automobilio modelį',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: _selectedModel == null
                            ? colors.mutedText
                            : colors.onSurface,
                      ),
                    ),
                  ),
                  const SizedBox(height: AppUiTokens.fieldVerticalSpacing),
                  _fieldLabel(context, 'Karta ir metai'),
                  const SizedBox(height: AppUiTokens.fieldLabelBottomSpacing),
                  _fieldShell(
                    context: context,
                    onTap: _selectedModel == null
                        ? null
                        : () async {
                            final generations = _generations(_fitments, _selectedModel);
                            if (generations.isEmpty) return;
                            final selectedGen = await showModalBottomSheet<VehicleFitment>(
                              context: context,
                              isScrollControlled: true,
                              useSafeArea: true,
                              builder: (sheetContext) => _buildGenerationPicker(sheetContext, generations),
                            );
                            if (selectedGen == null || !mounted) return;
                            setState(() {
                              _selectedGeneration = selectedGen;
                            });
                          },
                    showChevron: _selectedModel != null,
                    child: Text(
                      _selectedModel == null
                          ? 'Pasirinkite automobilio modelį'
                          : _selectedGeneration?.generationLabel ?? 'Pasirinkite kartą',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: _selectedModel == null || _selectedGeneration == null
                            ? colors.mutedText
                            : colors.onSurface,
                      ),
                    ),
                  ),
                  const Spacer(),
                  ActionButtonStyles.primaryButton(
                    context: context,
                    onPressed: _selectedGeneration == null
                        ? null
                        : () async {
                            await context.read<AppState>().setSelectedCar(
                              _selectedGeneration,
                            );
                            if (!context.mounted) return;
                            Navigator.of(context).pop();
                          },
                    label: 'Išsaugoti automobilį',
                  ),
                  if (context.watch<AppState>().selectedCar != null) ...[
                    const SizedBox(height: ActionButtonStyles.buttonStackSpacing),
                    ActionButtonStyles.tertiaryDestructiveButton(
                      context: context,
                      onPressed: () async {
                        await context.read<AppState>().setSelectedCar(null);
                        if (!context.mounted) return;
                        Navigator.of(context).pop();
                      },
                      label: 'Pašalinti pasirinktą automobilį',
                    ),
                  ],
                ],
              ),
            ),
    );
  }
}

extension<T> on Iterable<T> {
  T? get firstOrNull => isEmpty ? null : first;
}
