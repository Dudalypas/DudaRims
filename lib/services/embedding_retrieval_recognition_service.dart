import 'dart:io';
import 'package:flutter/foundation.dart';

import '../core/constants/app_constants.dart';
import '../ml/tflite_wheel_embedding_pipeline.dart';
import '../models/recognition_candidate.dart';
import '../models/recognition_outcome.dart';
import 'local_fitment_repository.dart';
import 'recognition_service.dart';
import 'wheel_crop_service.dart';
import 'wheel_detector_service.dart';

class EmbeddingRetrievalRecognitionService implements RecognitionService {
  final LocalFitmentRepository repository;
  final WheelDetectorService detector;
  final WheelCropService cropper;
  final TfliteWheelEmbeddingPipeline embeddingPipeline;
  final RecognitionService? fallbackService;

  EmbeddingRetrievalRecognitionService({
    required this.repository,
    WheelDetectorService? detector,
    WheelCropService? cropper,
    TfliteWheelEmbeddingPipeline? embeddingPipeline,
    this.fallbackService,
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
    }

    final wheelSpecs = await repository.loadWheelSpecs();

    final allowedClasses = <String>{
      for (final spec in wheelSpecs)
        if (spec.wheelClassName.trim().isNotEmpty)
          spec.wheelClassName.trim().toLowerCase(),
    };

    File retrievalInput = imageFile;
    try {
      try {
        final detectionRun = await detector.detectBest(
          imageFile,
          saveDebugImage: false,
        );
        if (detectionRun.bestDetection == null) {
          if (AppConstants.enablePipelineDebugLogs) {
            debugPrint(
              '[EmbeddingRetrieval] Detector produced no box, using original image for embedding.',
            );
          }
        } else {
          retrievalInput = await cropper.cropDetectedWheel(
            imageFile,
            detectionRun.bestDetection!,
          );
          if (AppConstants.enablePipelineDebugLogs) {
            debugPrint('[EmbeddingRetrieval] Crop refinement completed.');
          }
        }
      } catch (e, st) {
        if (AppConstants.enablePipelineDebugLogs) {
          debugPrint(
            '[EmbeddingRetrieval] Detector/crop step failed, using original image. error=$e',
          );
          debugPrint('$st');
        }
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
            '[EmbeddingRetrieval] No candidates after metadata filter, triggering fallback.',
          );
          final rawLabels = retrieved.take(5).map((e) => e.label).join(', ');
          debugPrint(
            '[EmbeddingRetrieval] Raw labels before filter: $rawLabels',
          );
        }
        return _fallbackOrFailure(
          imageFile,
          AppConstants.recognitionFailureMessage,
        );
      }

      candidates.sort((a, b) => b.score.compareTo(a.score));
      if (AppConstants.enablePipelineDebugLogs) {
        debugPrint(
          '[EmbeddingRetrieval] Success via embedding path. top1=${candidates.first.label} '
          'score=${candidates.first.score.toStringAsFixed(4)}',
        );
      }
      return RecognitionOutcome(
        top5: candidates.take(5).toList(growable: false),
      );
    } catch (e, st) {
      if (AppConstants.enablePipelineDebugLogs) {
        debugPrint('[EmbeddingRetrieval] Exception in embedding path: $e');
        debugPrint('$st');
      }
      return _fallbackOrFailure(
        imageFile,
        AppConstants.recognitionFailureMessage,
      );
    }
  }

  Future<RecognitionOutcome> _fallbackOrFailure(
    File imageFile,
    String reason,
  ) async {
    if (AppConstants.enableClassifierFallback && fallbackService != null) {
      try {
        if (AppConstants.enablePipelineDebugLogs) {
          debugPrint('[EmbeddingRetrieval] Entering classifier fallback path.');
        }
        return await fallbackService!.analyze(imageFile);
      } catch (e, st) {
        if (AppConstants.enablePipelineDebugLogs) {
          debugPrint('[EmbeddingRetrieval] Classifier fallback failed: $e');
          debugPrint('$st');
        }
        return RecognitionOutcome.failure(reason: reason);
      }
    }
    if (AppConstants.enablePipelineDebugLogs) {
      debugPrint('[EmbeddingRetrieval] Fallback disabled, returning failure.');
    }
    return RecognitionOutcome.failure(reason: reason);
  }
}
