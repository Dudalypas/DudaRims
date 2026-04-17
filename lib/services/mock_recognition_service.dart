import 'dart:io';
import 'dart:math';

import '../models/recognition_candidate.dart';
import '../models/recognition_outcome.dart';
import 'local_fitment_repository.dart';
import 'recognition_service.dart';

class MockRecognitionService implements RecognitionService {
  final LocalFitmentRepository repository;

  MockRecognitionService(this.repository);

  @override
  Future<RecognitionOutcome> analyze(File imageFile) async {
    // Mockas UI testams, jungikli pasidaryt reiktu
    final wheelSpecs = await repository.loadWheelSpecs();
    final classes = <String>{
      for (final spec in wheelSpecs)
        if (spec.wheelClassName.trim().isNotEmpty) spec.wheelClassName.trim(),
    }.toList(growable: false);

    if (classes.length < 5) {
      throw StateError('Nepakanka ratlankių klasių mock atpažinimui.');
    }

    final seed = imageFile.path.hashCode ^ DateTime.now().day;
    final random = Random(seed);

    classes.shuffle(random);
    final picked = classes.take(5).toList(growable: false);

    final top1High = random.nextBool();
    final baseTop = top1High ? 0.86 : 0.64;

    final candidates = <RecognitionCandidate>[];
    for (var i = 0; i < picked.length; i++) {
      final scoreDrop = i * (top1High ? 0.09 : 0.06);
      final jitter = random.nextDouble() * 0.035;
      final score = (baseTop - scoreDrop - jitter).clamp(0.05, 0.98).toDouble();
      candidates.add(RecognitionCandidate(label: picked[i], score: score));
    }

    candidates.sort((a, b) => b.score.compareTo(a.score));
    return RecognitionOutcome(top5: candidates);
  }
}
