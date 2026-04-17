import 'dart:io';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'prediction.dart';
import 'wheel_pipeline.dart';

class TfliteWheelPipeline implements WheelPipeline {
  final String modelAsset;
  final String labelsAsset;

  Interpreter? _interpreter;
  List<String>? _labels;
  List<int>? _inputShape; // Tipiskai [1,224,224,3], bet imam dinamiskai is modelio.
  List<int>? _outputShape; // Dažniausiai [1,numClasses].
  Future<void>? _loadFuture;

  TfliteWheelPipeline({
    this.modelAsset =
        'assets/models/wheel_classifier_cropped_best_float32v4.tflite',
    this.labelsAsset = 'assets/labels/labels.txt',
  });

  Future<void> _ensureLoaded() {
    _loadFuture ??= _load();
    return _loadFuture!;
  }

  Future<void> _load() async {
    _interpreter = await Interpreter.fromAsset(modelAsset);

    final labelsRaw = await rootBundle.loadString(labelsAsset);
    _labels = labelsRaw
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList();

    _inputShape = _interpreter!.getInputTensor(0).shape;
    _outputShape = _interpreter!.getOutputTensor(0).shape;

    debugPrint(
      'Input: $_inputShape | Output: $_outputShape | Labels: ${_labels!.length}',
    );
  }

  @override
  Future<List<Prediction>> predict(File imageFile) async {
    await _ensureLoaded();
    final interpreter = _interpreter!;
    final labels = _labels!;
    final inputShape = _inputShape!;
    final outputShape = _outputShape!;

    // Pirma dekoduojam faila i image objekta
    final decoded = img.decodeImage(await imageFile.readAsBytes());
    if (decoded == null) throw Exception('Nepavyko dekoduoti paveikslėlio');

    // Imame target dydi is input tensoriaus, kad nereiktu hardcodinti 224
    final h = inputShape[1];
    final w = inputShape[2];

    // Sutvarkom EXIF orientacija, nes kitaip dalis telefonu fotkiu ateina pasuktos
    final oriented = img.bakeOrientation(
      decoded,
    );
    // Resize darom tiesiai i modelio dydi
    final resized = img.copyResize(oriented, width: w, height: h);

    // Siame projekte modelis treniruotas su float32 RGB [0..255], tai papildomai nenormalizuojam
    // Jei ka paskui pakeist modeli i toki, kuris treniruotas su [0..1] floatais, tai cia reikes padalinti is 255.0
    final input = Float32List(1 * h * w * 3);
    var i = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final px = resized.getPixel(x, y);
        input[i++] = px.r.toDouble();
        input[i++] = px.g.toDouble();
        input[i++] = px.b.toDouble();
      }
    }
    final inputTensor = input.reshape([1, h, w, 3]);

    // Is modelio gaunam logits [1,numClasses], tada rankinimui taikom softmax
    final numClasses = outputShape.last;
    final output = List.generate(1, (_) => List.filled(numClasses, 0.0));

    interpreter.run(inputTensor, output);

    final logits = output[0];
    final probs = _softmax(logits);

    final top = _topK(probs, k: min(5, probs.length));

    return top.map((e) {
      final idx = e.$1;
      final score = e.$2;
      final label = (idx < labels.length) ? labels[idx] : 'class_$idx';
      return Prediction(label, score);
    }).toList();
  }

  List<double> _softmax(List<double> v) {
    final m = v.reduce(max);
    final exps = v.map((x) => exp(x - m)).toList();
    final s = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / s).toList();
  }

  List<(int, double)> _topK(List<double> probs, {required int k}) {
    final idx = List.generate(probs.length, (i) => i);
    idx.sort((a, b) => probs[b].compareTo(probs[a]));
    return idx.take(k).map((i) => (i, probs[i])).toList();
  }

  void dispose() {
    _interpreter?.close();
  }
}
