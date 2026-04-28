import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../core/constants/app_constants.dart';
import '../ml/tflite_wheel_embedding_pipeline.dart';
import '../models/recognition_candidate.dart';
import '../models/recognition_outcome.dart';
import 'local_fitment_repository.dart';
import 'recognition_service.dart';
import 'wheel_crop_service.dart';
import 'wheel_detector_service.dart';

// This app uses embedding retrieval for runtime recognition. Legacy classifier
// fallback was removed from runtime to keep production pipeline deterministic
// and thesis-aligned.
class EmbeddingRetrievalRecognitionService implements RecognitionService {
  final LocalFitmentRepository repository;
  final WheelDetectorService detector;
  final WheelCropService cropper;
  final TfliteWheelEmbeddingPipeline embeddingPipeline;

  EmbeddingRetrievalRecognitionService({
    required this.repository,
    WheelDetectorService? detector,
    WheelCropService? cropper,
    TfliteWheelEmbeddingPipeline? embeddingPipeline,
  }) : detector = detector ?? WheelDetectorService(),
       cropper =
           cropper ??
           const WheelCropService(
             paddingRatio: AppConstants.retrievalCropPaddingRatio,
             tightenRatio: AppConstants.retrievalCropTightenRatio,
             enforceSquare: AppConstants.retrievalCropEnforceSquare,
           ),
       embeddingPipeline =
           embeddingPipeline ??
           TfliteWheelEmbeddingPipeline(
             modelAsset: AppConstants.embeddingModelAsset,
             referenceEmbeddingsAsset: AppConstants.referenceEmbeddingsAsset,
             referenceMode: AppConstants.retrievalReferenceMode,
             aggregationMode: AppConstants.classAggregationMode,
             aggregationTopN: AppConstants.classAggregationTopN,
           );

  @override
  Future<RecognitionOutcome> analyze(File imageFile) async {
    if (AppConstants.enablePipelineDebugLogs) {
      debugPrint('[EmbeddingRetrieval] Analyze started.');
      debugPrint('[EmbeddingRetrieval] Active assets: model=${embeddingPipeline.modelAsset} refs=${embeddingPipeline.referenceEmbeddingsAsset} mode=${embeddingPipeline.referenceMode}');
      debugPrint(
        '[EmbeddingRetrieval] Ellipse mask enabled=${AppConstants.enableRetrievalEllipseMask} '
        'inset=${AppConstants.retrievalEllipseMaskInsetRatio.toStringAsFixed(3)} '
        'feather=${AppConstants.retrievalEllipseMaskFeather.toStringAsFixed(3)} '
        'circleFallback=${AppConstants.retrievalEllipseMaskUseCircleFallback}',
      );
    }

    final wheelSpecs = await repository.loadWheelSpecs();

    final allowedClasses = <String>{
      for (final spec in wheelSpecs)
        if (spec.wheelClassName.trim().isNotEmpty)
          spec.wheelClassName.trim().toLowerCase(),
    };

    final sourceImage = img.decodeImage(await imageFile.readAsBytes());
    if (sourceImage == null) {
      return RecognitionOutcome.failure(
        reason: AppConstants.recognitionFailureMessage,
      );
    }

    final orientedSource = img.bakeOrientation(sourceImage);

    File retrievalInput = imageFile;
    final tempFilesToDelete = <File>[];
    try {
      bool useCenteredRimOnlyFallback = false;
      bool usedMaskedCropPath = false;
      try {
        final detectionRun = await detector.detectBest(
          imageFile,
          saveDebugImage: false,
        );
        if (detectionRun.bestDetection == null) {
          if (AppConstants.enablePipelineDebugLogs) {
            debugPrint(
              '[EmbeddingRetrieval][reject] reason=detector_no_box',
            );
          }
          return RecognitionOutcome.failure(
            reason: AppConstants.recognitionNoRimDetectedMessage,
          );
        }

        final detection = detectionRun.bestDetection!;
        final boxWidth = detection.width;
        final boxHeight = detection.height;
        final imageArea = (orientedSource.width * orientedSource.height)
          .toDouble();
        final boxArea = boxWidth * boxHeight;
        final areaRatio = imageArea > 0 ? boxArea / imageArea : 0.0;
        final longSide = boxWidth > boxHeight ? boxWidth : boxHeight;
        final shortSide = boxWidth > boxHeight ? boxHeight : boxWidth;
        final aspectRatio = shortSide > 0 ? longSide / shortSide : 0.0;

        final hardScoreReject =
          detection.score < AppConstants.detectorHardMinScoreThreshold;
        final hardTinyReject = areaRatio < AppConstants.detectorHardMinBboxAreaRatio;
        final hardAspectReject =
          aspectRatio > AppConstants.detectorHardMaxAspectRatio;

        if (hardScoreReject || hardTinyReject || hardAspectReject) {
          if (AppConstants.enablePipelineDebugLogs) {
            if (hardScoreReject) {
              debugPrint(
                '[EmbeddingRetrieval][reject] reason=detector_score_too_low score=${detection.score.toStringAsFixed(4)} min=${AppConstants.detectorHardMinScoreThreshold.toStringAsFixed(4)}',
              );
            } else {
              debugPrint(
                '[EmbeddingRetrieval][reject] reason=detector_box_invalid areaRatio=${areaRatio.toStringAsFixed(4)} aspectRatio=${aspectRatio.toStringAsFixed(4)}',
              );
            }
          }
          return RecognitionOutcome.failure(
            reason: AppConstants.recognitionNoRimDetectedMessage,
          );
        }

        final borderlineScore =
            detection.score < AppConstants.detectorMinScoreThreshold;
        final borderlineArea =
            areaRatio < AppConstants.detectorMinBboxAreaRatio ||
            areaRatio > AppConstants.detectorMaxBboxAreaRatio;
        final borderlineAspect =
            aspectRatio > AppConstants.detectorMaxAspectRatio;

        if (borderlineScore || borderlineArea || borderlineAspect) {
          if (AppConstants.enablePipelineDebugLogs) {
            if (borderlineScore) {
              debugPrint(
                '[EmbeddingRetrieval][reject] reason=detector_score_too_low score=${detection.score.toStringAsFixed(4)} min=${AppConstants.detectorMinScoreThreshold.toStringAsFixed(4)}',
              );
            }
            if (borderlineArea || borderlineAspect) {
              debugPrint(
                '[EmbeddingRetrieval][reject] reason=detector_box_invalid areaRatio=${areaRatio.toStringAsFixed(4)} aspectRatio=${aspectRatio.toStringAsFixed(4)}',
              );
            }
          }
          useCenteredRimOnlyFallback = true;
        } else {
          final refinedCrop = await cropper.cropDetectedWheel(
            imageFile,
            detection,
          );
          if (AppConstants.enablePipelineDebugLogs) {
            debugPrint('[EmbeddingRetrieval] Crop refinement completed.');
          }
          final processed = await _applyOptionalEllipseMask(refinedCrop);
          retrievalInput = processed.file;
          usedMaskedCropPath = processed.usedMask;
          if (processed.file.path != refinedCrop.path) {
            tempFilesToDelete.add(processed.file);
          }
        }
      } catch (e, st) {
        if (AppConstants.enablePipelineDebugLogs) {
          debugPrint(
            '[EmbeddingRetrieval] Detector/crop step failed. error=$e',
          );
          debugPrint('$st');
        }
        return RecognitionOutcome.failure(
          reason: AppConstants.recognitionNoRimDetectedMessage,
        );
      }

      if (useCenteredRimOnlyFallback) {
        final fallbackOutcome = await _tryCenteredRimOnlyFallback(
          imageFile: imageFile,
          orientedSource: orientedSource,
          allowedClasses: allowedClasses,
        );
        if (fallbackOutcome != null) {
          return fallbackOutcome;
        }
        return RecognitionOutcome.failure(
          reason: AppConstants.recognitionNoRimDetectedMessage,
        );
      }

      final retrieved = await embeddingPipeline.retrieveTopK(
        retrievalInput,
        k: 5,
      );

      if (AppConstants.enablePipelineDebugLogs) {
        debugPrint(
          '[EmbeddingRetrieval] Retrieved raw candidates: ${retrieved.length}.',
        );
        if (retrieved.isNotEmpty) {
          final preview = retrieved
              .take(3)
              .map((r) => '${r.label}:${r.cosineSimilarity.toStringAsFixed(3)}')
              .join(', ');
          debugPrint('[EmbeddingRetrieval] Raw top-3: $preview');
        }
      }

      if (retrieved.isEmpty) {
        if (AppConstants.enablePipelineDebugLogs) {
          debugPrint(
            '[EmbeddingRetrieval][reject] reason=retrieval_similarity_too_low top1=none min=${AppConstants.retrievalMinTop1Similarity.toStringAsFixed(4)}',
          );
        }
        return RecognitionOutcome.failure(
          reason: AppConstants.recognitionUncertainResultMessage,
        );
      }

      final candidates = <RecognitionCandidate>[];
      for (final r in retrieved) {
        final label = r.label.trim();
        if (label.isEmpty) continue;
        if (!allowedClasses.contains(label.toLowerCase())) continue;
        candidates.add(
          RecognitionCandidate(label: label, score: r.normalizedScore),
        );
      }

      if (candidates.isEmpty) {
        if (AppConstants.enablePipelineDebugLogs) {
          debugPrint(
            '[EmbeddingRetrieval] No candidates after metadata filter.',
          );
          final rawLabels = retrieved.take(5).map((e) => e.label).join(', ');
          debugPrint(
            '[EmbeddingRetrieval] Raw labels before filter: $rawLabels',
          );
        }
        return RecognitionOutcome.failure(
          reason: AppConstants.recognitionUncertainResultMessage,
        );
      }

      candidates.sort((a, b) => b.score.compareTo(a.score));
      final manualTop3 = candidates.take(3).toList(growable: false);

      final top1Similarity = retrieved.first.cosineSimilarity;
      if (top1Similarity < AppConstants.retrievalMinTop1Similarity) {
        if (AppConstants.enablePipelineDebugLogs) {
          final preview = manualTop3
              .map((c) => '${c.label}:${c.score.toStringAsFixed(3)}')
              .join(', ');
          debugPrint(
            '[EmbeddingRetrieval][manual_pick_low_confidence] top1=${top1Similarity.toStringAsFixed(4)} min=${AppConstants.retrievalMinTop1Similarity.toStringAsFixed(4)} top3=$preview',
          );
        }
        return RecognitionOutcome.manualPick(
          topCandidates: manualTop3,
          reason: RecognitionManualPickReason.lowTop1Similarity,
        );
      }

      if (retrieved.length >= 2) {
        final top2Similarity = retrieved[1].cosineSimilarity;
        final margin = top1Similarity - top2Similarity;
        if (margin < AppConstants.retrievalMinTop1Top2Margin) {
          if (AppConstants.enablePipelineDebugLogs) {
            final preview = manualTop3
                .map((c) => '${c.label}:${c.score.toStringAsFixed(3)}')
                .join(', ');
            debugPrint(
              '[EmbeddingRetrieval][manual_pick_small_margin] margin=${margin.toStringAsFixed(4)} min=${AppConstants.retrievalMinTop1Top2Margin.toStringAsFixed(4)} top1=${top1Similarity.toStringAsFixed(4)} top2=${top2Similarity.toStringAsFixed(4)} top3=$preview',
            );
          }
          return RecognitionOutcome.manualPick(
            topCandidates: manualTop3,
            reason: RecognitionManualPickReason.smallTop12Margin,
          );
        }
      }

      if (AppConstants.enablePipelineDebugLogs) {
        debugPrint(
          '[EmbeddingRetrieval] Success via embedding path. top1=${candidates.first.label} '
          'score=${candidates.first.score.toStringAsFixed(4)}',
        );
        if (usedMaskedCropPath) {
          debugPrint('[EmbeddingRetrieval] Success via masked embedding path.');
        }
      }
      return RecognitionOutcome(
        top5: candidates.take(5).toList(growable: false),
      );
    } catch (e, st) {
      if (AppConstants.enablePipelineDebugLogs) {
        debugPrint('[EmbeddingRetrieval] Exception in embedding path: $e');
        debugPrint('$st');
      }
      return RecognitionOutcome.failure(
        reason: AppConstants.recognitionFailureMessage,
      );
    } finally {
      for (final file in tempFilesToDelete) {
        if (await file.exists()) {
          await file.delete();
        }
      }
    }
  }

  Future<_MaskingResult> _applyOptionalEllipseMask(File refinedCropFile) async {
    if (!AppConstants.enableRetrievalEllipseMask) {
      return _MaskingResult(file: refinedCropFile, usedMask: false);
    }

    final decoded = img.decodeImage(await refinedCropFile.readAsBytes());
    if (decoded == null) {
      return _MaskingResult(file: refinedCropFile, usedMask: false);
    }

    final oriented = img.bakeOrientation(decoded);
    final width = oriented.width;
    final height = oriented.height;
    if (AppConstants.enablePipelineDebugLogs) {
      debugPrint('[EmbeddingRetrieval] Crop size before masking: ${width}x$height');
    }

    final cx = (width - 1) / 2.0;
    final cy = (height - 1) / 2.0;
    final inset = AppConstants.retrievalEllipseMaskInsetRatio.clamp(0.0, 0.45);
    final feather = AppConstants.retrievalEllipseMaskFeather.clamp(0.0, 0.4);

    final rxBase = width * 0.5 * (1.0 - inset);
    final ryBase = height * 0.5 * (1.0 - inset);
    final useCircle = AppConstants.retrievalEllipseMaskUseCircleFallback;
    final rx = useCircle ? math.min(rxBase, ryBase) : rxBase;
    final ry = useCircle ? math.min(rxBase, ryBase) : ryBase;

    final out = img.Image.from(oriented);
    final inner = (1.0 - feather).clamp(0.0, 1.0);

    for (var y = 0; y < height; y++) {
      final ny = (y - cy) / (ry <= 1e-6 ? 1.0 : ry);
      for (var x = 0; x < width; x++) {
        final nx = (x - cx) / (rx <= 1e-6 ? 1.0 : rx);
        final d = math.sqrt(nx * nx + ny * ny);

        double keep;
        if (d <= inner) {
          keep = 1.0;
        } else if (d >= 1.0) {
          keep = 0.0;
        } else {
          final t = (d - inner) / math.max(1e-6, (1.0 - inner));
          keep = 1.0 - t;
        }

        if (keep >= 0.999) continue;

        final px = out.getPixel(x, y);
        out.setPixelRgb(
          x,
          y,
          (px.r * keep).round().clamp(0, 255),
          (px.g * keep).round().clamp(0, 255),
          (px.b * keep).round().clamp(0, 255),
        );
      }
    }

    final maskedFile = File(
      '${Directory.systemTemp.path}${Platform.pathSeparator}retrieval_masked_${DateTime.now().microsecondsSinceEpoch}.jpg',
    );
    await maskedFile.writeAsBytes(img.encodeJpg(out, quality: 95));

    if (AppConstants.enablePipelineDebugLogs) {
      debugPrint(
        '[EmbeddingRetrieval] Masked crop path used. mode=${useCircle ? 'circle' : 'ellipse'}',
      );
    }

    return _MaskingResult(file: maskedFile, usedMask: true);
  }

  Future<RecognitionOutcome?> _tryCenteredRimOnlyFallback({
    required File imageFile,
    required img.Image orientedSource,
    required Set<String> allowedClasses,
  }) async {
    if (!AppConstants.enableCenteredRimOnlyFallback) {
      return null;
    }

    if (!_appearsDominantCenteredObject(orientedSource)) {
      if (AppConstants.enablePipelineDebugLogs) {
        debugPrint(
          '[EmbeddingRetrieval][reject] reason=detector_no_box dominant_centered_object=false',
        );
      }
      return null;
    }

    final maxCropSide = math.min(orientedSource.width, orientedSource.height);
    final cropSize = ((maxCropSide *
          AppConstants.rimOnlyFallbackCenteredCropRatio)
        .round()
        .clamp(32, maxCropSide))
      .toInt();
    final left = ((orientedSource.width - cropSize) / 2).round();
    final top = ((orientedSource.height - cropSize) / 2).round();

    final centered = img.copyCrop(
      orientedSource,
      x: left,
      y: top,
      width: cropSize,
      height: cropSize,
    );

    final tempFile = File(
      '${Directory.systemTemp.path}${Platform.pathSeparator}rim_only_centered_${DateTime.now().microsecondsSinceEpoch}.jpg',
    );
    try {
      await tempFile.writeAsBytes(img.encodeJpg(centered, quality: 95));

      if (AppConstants.enablePipelineDebugLogs) {
        debugPrint(
          '[EmbeddingRetrieval] Using centered rim-only fallback crop. size=${cropSize}x$cropSize',
        );
      }

      final retrieved = await embeddingPipeline.retrieveTopK(tempFile, k: 5);
      if (retrieved.isEmpty) {
        if (AppConstants.enablePipelineDebugLogs) {
          debugPrint(
            '[EmbeddingRetrieval][reject] reason=retrieval_similarity_too_low top1=none min=${AppConstants.rimOnlyFallbackMinTop1Similarity.toStringAsFixed(4)} fallback=rim_only_centered',
          );
        }
        return null;
      }

      final top1 = retrieved.first.cosineSimilarity;
      if (top1 < AppConstants.rimOnlyFallbackMinTop1Similarity) {
        if (AppConstants.enablePipelineDebugLogs) {
          debugPrint(
            '[EmbeddingRetrieval][reject] reason=retrieval_similarity_too_low top1=${top1.toStringAsFixed(4)} min=${AppConstants.rimOnlyFallbackMinTop1Similarity.toStringAsFixed(4)} fallback=rim_only_centered',
          );
        }
        return null;
      }

      if (retrieved.length >= 2) {
        final margin = top1 - retrieved[1].cosineSimilarity;
        if (margin < AppConstants.rimOnlyFallbackMinTop1Top2Margin) {
          if (AppConstants.enablePipelineDebugLogs) {
            debugPrint(
              '[EmbeddingRetrieval][reject] reason=retrieval_margin_too_small margin=${margin.toStringAsFixed(4)} min=${AppConstants.rimOnlyFallbackMinTop1Top2Margin.toStringAsFixed(4)} fallback=rim_only_centered',
            );
          }
          return null;
        }
      }

      final candidates = <RecognitionCandidate>[];
      for (final r in retrieved) {
        final label = r.label.trim();
        if (label.isEmpty) continue;
        if (!allowedClasses.contains(label.toLowerCase())) continue;
        candidates.add(
          RecognitionCandidate(label: label, score: r.normalizedScore),
        );
      }

      if (candidates.isEmpty) {
        return null;
      }

      candidates.sort((a, b) => b.score.compareTo(a.score));
      if (AppConstants.enablePipelineDebugLogs) {
        debugPrint(
          '[EmbeddingRetrieval] Success via centered rim-only fallback. top1=${candidates.first.label} score=${candidates.first.score.toStringAsFixed(4)}',
        );
      }
      return RecognitionOutcome(top5: candidates.take(5).toList(growable: false));
    } finally {
      if (await tempFile.exists()) {
        await tempFile.delete();
      }
    }
  }

  bool _appearsDominantCenteredObject(img.Image image) {
    final resized = img.copyResize(image, width: 192, height: 192);
    final centerMin = (192 * 0.25).round();
    final centerMax = (192 * 0.75).round();

    var totalEnergy = 0.0;
    var centerEnergy = 0.0;

    for (var y = 1; y < 191; y++) {
      for (var x = 1; x < 191; x++) {
        final p = resized.getPixel(x, y);
        final px = resized.getPixel(x + 1, y);
        final py = resized.getPixel(x, y + 1);

        final dx = (p.r - px.r).abs() + (p.g - px.g).abs() + (p.b - px.b).abs();
        final dy = (p.r - py.r).abs() + (p.g - py.g).abs() + (p.b - py.b).abs();
        final e = (dx + dy).toDouble();

        totalEnergy += e;
        if (x >= centerMin && x <= centerMax && y >= centerMin && y <= centerMax) {
          centerEnergy += e;
        }
      }
    }

    if (totalEnergy <= 1e-6) {
      return false;
    }

    final centerRatio = centerEnergy / totalEnergy;
    return centerRatio >= AppConstants.rimOnlyFallbackCenterEnergyMinRatio;
  }

}

class _MaskingResult {
  final File file;
  final bool usedMask;

  const _MaskingResult({required this.file, required this.usedMask});
}
