import 'dart:io';

import '../models/recognition_outcome.dart';

abstract class RecognitionService {
  Future<RecognitionOutcome> analyze(File imageFile);
}
