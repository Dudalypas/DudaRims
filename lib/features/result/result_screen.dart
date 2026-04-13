import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../app/state/app_state.dart';
import '../../ml/prediction.dart';
import '../../models/detection_result.dart';
import '../../models/rim_model.dart';
import '../../services/rim_catalog_service.dart';
import '../rim/rim_detail_screen.dart';

class ResultScreen extends StatefulWidget {
  final File imageFile;
  final File? classifierInputImageFile;
  final File? detectorDebugImageFile;
  final DetectionResult? detection;
  final String? pipelineMessage;
  final List<Prediction> predictions;

  const ResultScreen({
    super.key,
    required this.imageFile,
    this.classifierInputImageFile,
    this.detectorDebugImageFile,
    this.detection,
    this.pipelineMessage,
    required this.predictions,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  final _catalogService = RimCatalogService();
  Prediction? _selectedCandidate;

  void _openDetails(Prediction prediction) {
    final appState = context.read<AppState>();
    final rim = _catalogService.resolveByClass(prediction.label, appState.wheelSpecs);
    if (rim == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Šio ratlankio specifikacija duomenų bazėje nerasta.')),
      );
      return;
    }

    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => RimDetailScreen(
          previewImage: widget.classifierInputImageFile ?? widget.imageFile,
          rimModel: rim,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final appState = context.watch<AppState>();
    final predictions = widget.predictions;

    if (predictions.isEmpty) {
      return Scaffold(
        appBar: AppBar(title: const Text('Atpažinimo rezultatas')),
        body: const Center(child: Text('Modelis negrąžino rezultatų.')),
      );
    }

    final top = predictions.first;
    final highConfidence = top.score >= AppState.confidenceThreshold;
    final candidates = highConfidence ? [top] : predictions.take(3).toList(growable: false);
    final activeCandidate = highConfidence ? top : _selectedCandidate;

    return Scaffold(
      appBar: AppBar(title: const Text('Atpažinimo rezultatas')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if ((widget.pipelineMessage ?? '').isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Text(widget.pipelineMessage!, style: const TextStyle(fontWeight: FontWeight.w600)),
            ),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.file(
                      widget.classifierInputImageFile ?? widget.imageFile,
                      width: 86,
                      height: 86,
                      fit: BoxFit.cover,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          highConfidence ? 'Rastas vienas aiškus rezultatas' : 'Galimi ratlankio variantai',
                          style: const TextStyle(fontWeight: FontWeight.w700),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          highConfidence
                              ? 'Tikėtinas modelis: ${top.label}'
                              : 'Pasirink labiausiai panašų variantą.',
                        ),
                        const SizedBox(height: 6),
                        Text(
                          'Slenkstis: ${(AppState.confidenceThreshold * 100).toStringAsFixed(0)}%',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          if (highConfidence)
            _HeroResultCard(
              prediction: top,
              rim: _catalogService.resolveByClass(top.label, appState.wheelSpecs),
              onOpen: () => _openDetails(top),
            )
          else
            SizedBox(
              height: 212,
              child: ListView.separated(
                scrollDirection: Axis.horizontal,
                itemCount: candidates.length,
                separatorBuilder: (_, _) => const SizedBox(width: 10),
                itemBuilder: (context, index) {
                  final c = candidates[index];
                  final selected = c == _selectedCandidate;
                  final rim = _catalogService.resolveByClass(c.label, appState.wheelSpecs);
                  return _CandidateCard(
                    prediction: c,
                    rim: rim,
                    imageFile: widget.classifierInputImageFile ?? widget.imageFile,
                    selected: selected,
                    onTap: () => setState(() => _selectedCandidate = c),
                  );
                },
              ),
            ),
          const SizedBox(height: 12),
          if (!highConfidence)
            FilledButton(
              onPressed: activeCandidate == null ? null : () => _openDetails(activeCandidate),
              child: const Text('Tęsti su pasirinktu variantu'),
            ),
          const SizedBox(height: 10),
          if (appState.selectedCar == null)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: const [
                    Icon(Icons.info_outline),
                    SizedBox(width: 8),
                    Expanded(child: Text('Pasirink automobilį, kad matytum suderinamumą detalėse.')),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _HeroResultCard extends StatelessWidget {
  final Prediction prediction;
  final RimModel? rim;
  final VoidCallback onOpen;

  const _HeroResultCard({
    required this.prediction,
    required this.rim,
    required this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              rim?.displayName ?? prediction.label,
              style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 6),
            Text(
              'Tikimybė ${(prediction.score * 100).toStringAsFixed(1)}%',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 12),
            FilledButton(onPressed: onOpen, child: const Text('Peržiūrėti specifikaciją ir suderinamumą')),
          ],
        ),
      ),
    );
  }
}

class _CandidateCard extends StatelessWidget {
  final Prediction prediction;
  final RimModel? rim;
  final File imageFile;
  final bool selected;
  final VoidCallback onTap;

  const _CandidateCard({
    required this.prediction,
    required this.rim,
    required this.imageFile,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(18),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        width: 180,
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(18),
          color: selected ? cs.primaryContainer.withValues(alpha: 0.55) : cs.surfaceContainerHighest.withValues(alpha: 0.4),
          border: Border.all(color: selected ? cs.primary : cs.outlineVariant),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.file(imageFile, width: 160, height: 100, fit: BoxFit.cover),
            ),
            const SizedBox(height: 10),
            Text(rim?.displayName ?? prediction.label, style: const TextStyle(fontWeight: FontWeight.w700)),
            const SizedBox(height: 4),
            Text('Tikimybė ${(prediction.score * 100).toStringAsFixed(1)}%', style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
      ),
    );
  }
}
