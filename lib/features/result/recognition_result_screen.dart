import 'dart:io';

import 'package:flutter/material.dart';

import '../../core/constants/rim_thumbnail_assets.dart';
import '../../core/theme/theme_tokens.dart';
import '../../models/recognition_candidate.dart';
import '../../models/recognition_outcome.dart';
import '../rim/rim_details_screen.dart';

class RecognitionResultScreen extends StatelessWidget {
  final File imageFile;
  final RecognitionOutcome outcome;

  const RecognitionResultScreen({
    super.key,
    required this.imageFile,
    required this.outcome,
  });

  @override
  Widget build(BuildContext context) {
    final top = outcome.top1;

    return Scaffold(
      appBar: AppBar(title: const Text('Atpažinimo rezultatas')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: Image.file(
                  imageFile,
                  height: 190,
                  width: double.infinity,
                  fit: BoxFit.cover,
                ),
              ),
              const SizedBox(height: 16),
              Text(
                outcome.isConfident
                    ? 'Labiausiai panašus ratlankis'
                    : 'Parink labiausiai panašų modelį',
                style: Theme.of(
                  context,
                ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 6),
              Text(
                outcome.isConfident
                    ? 'Atpažinimo atitikimas: ${(top.confidence * 100).toStringAsFixed(1)}%'
                    : 'Atpažinimo atitikimas žemesnis, pasirink vieną iš galimų variantų.',
                style: TextStyle(
                  color: Theme.of(context).colorScheme.mutedText,
                ),
              ),
              const SizedBox(height: 10),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.panelSurface,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  'Tai tik ratlankio atpažinimo rezultatas. Suderinamumas su automobiliu bus tikrinamas kitame žingsnyje.',
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.mutedText,
                  ),
                ),
              ),
              const SizedBox(height: 14),
              if (outcome.isConfident)
                _HeroCandidateCard(
                  candidate: top,
                  onContinue: () => _openDetails(context, top.label),
                )
              else
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: [
                      for (
                        var index = 0;
                        index < outcome.top3.length;
                        index++
                      ) ...[
                        if (index > 0) const SizedBox(width: 12),
                        _CandidateCard(
                          candidate: outcome.top3[index],
                          rank: index,
                          onTap: () =>
                              _openDetails(context, outcome.top3[index].label),
                        ),
                      ],
                    ],
                  ),
                ),
              const SizedBox(height: 16),
              const Text(
                'Top 5 kandidatai',
                style: TextStyle(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: outcome.top5
                    .map(
                      (c) => Chip(
                        label: Text(
                          '${c.label} • ${(c.confidence * 100).toStringAsFixed(0)}%',
                        ),
                      ),
                    )
                    .toList(growable: false),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _openDetails(BuildContext context, String className) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => RimDetailsScreen(
          imageFile: imageFile,
          selectedClassName: className,
        ),
      ),
    );
  }
}

class _HeroCandidateCard extends StatelessWidget {
  final RecognitionCandidate candidate;
  final VoidCallback onContinue;

  const _HeroCandidateCard({required this.candidate, required this.onContinue});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              candidate.label,
              style: Theme.of(
                context,
              ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            Text(
              'Atpažinimo atitikimas ${(candidate.confidence * 100).toStringAsFixed(1)}%',
              style: TextStyle(color: Theme.of(context).colorScheme.mutedText),
            ),
            const SizedBox(height: 14),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: onContinue,
                child: const Text('Peržiūrėti specifikaciją ir suderinamumą'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CandidateCard extends StatelessWidget {
  final RecognitionCandidate candidate;
  final int rank;
  final VoidCallback onTap;

  const _CandidateCard({
    required this.candidate,
    required this.rank,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final assetPath = resolveRimThumbnailAsset(candidate.label);

    final title = switch (rank) {
      0 => 'Labiausiai panašus',
      1 => 'Kitas galimas variantas',
      _ => 'Dar vienas galimas variantas',
    };

    return SizedBox(
      width: 236,
      child: Card(
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: AspectRatio(
                    aspectRatio: 4 / 3,
                    child: assetPath == null
                        ? Container(
                            color: colors.panelSurfaceHigh,
                            child: const Center(
                              child: Icon(Icons.album_rounded, size: 42),
                            ),
                          )
                        : Image.asset(
                            assetPath,
                            fit: BoxFit.cover,
                            errorBuilder: (context, error, stackTrace) {
                              return Container(
                                color: colors.panelSurfaceHigh,
                                child: const Center(
                                  child: Icon(Icons.album_rounded, size: 42),
                                ),
                              );
                            },
                          ),
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: colors.mutedText,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  candidate.label,
                  style: const TextStyle(fontWeight: FontWeight.w700),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 6),
                Text(
                  'Atpažinimo atitikimas ${(candidate.confidence * 100).toStringAsFixed(1)}%',
                  style: TextStyle(color: colors.mutedText),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
