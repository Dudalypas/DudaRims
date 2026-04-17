import 'dart:io';
import 'package:flutter/foundation.dart';

import '../core/constants/app_constants.dart';
import '../ml/tflite_wheel_pipeline.dart';
import '../models/recognition_candidate.dart';
import '../models/recognition_outcome.dart';
import 'local_fitment_repository.dart';
import 'recognition_service.dart';
import 'wheel_crop_service.dart';
import 'wheel_detector_service.dart';

class TfliteRecognitionService implements RecognitionService {
  final LocalFitmentRepository repository;
  final WheelDetectorService detector;
  final WheelCropService cropper;
  final TfliteWheelPipeline classifier;

  TfliteRecognitionService({
    required this.repository,
    WheelDetectorService? detector,
    WheelCropService? cropper,
    TfliteWheelPipeline? classifier,
  }) : detector = detector ?? WheelDetectorService(),
       cropper = cropper ?? const WheelCropService(),
       classifier = classifier ?? TfliteWheelPipeline();

  @override
  Future<RecognitionOutcome> analyze(File imageFile) async {
    if (AppConstants.enablePipelineDebugLogs) {
      debugPrint('[ClassifierBaseline] Analyze started.');
    }

    final wheelSpecs = await repository.loadWheelSpecs();

    File? classifierInput = imageFile;
    try {
      final detectionRun = await detector.detectBest(
        imageFile,
        saveDebugImage: false,
      );
      if (detectionRun.bestDetection == null) {
        return RecognitionOutcome.failure(
          reason: AppConstants.recognitionFailureMessage,
        );
      }

      classifierInput = await cropper.cropDetectedWheel(
        imageFile,
        detectionRun.bestDetection!,
      );
    } catch (_) {
      classifierInput = imageFile;
    }

    final predictions = await classifier.predict(classifierInput);

    final allowedClasses = <String>{
      for (final spec in wheelSpecs)
        if (spec.wheelClassName.trim().isNotEmpty)
          spec.wheelClassName.trim().toLowerCase(),
    };

    final candidates = <RecognitionCandidate>[];
    for (final p in predictions) {
      final label = p.label.trim();
      if (label.isEmpty) continue;
      if (!allowedClasses.contains(label.toLowerCase())) continue;
      candidates.add(RecognitionCandidate(label: label, score: p.score));
    }

    if (candidates.isEmpty) {
      return RecognitionOutcome.failure(
        reason: AppConstants.recognitionFailureMessage,
      );
    }

    candidates.sort((a, b) => b.score.compareTo(a.score));
    final top5 = candidates.take(5).toList(growable: false);

    return RecognitionOutcome(top5: top5);
  }
}
