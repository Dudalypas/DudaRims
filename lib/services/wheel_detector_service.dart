import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/detection_result.dart';

class WheelDetectionRunResult {
  final DetectionResult? bestDetection;
  final File? debugImageFile;

  const WheelDetectionRunResult({
    required this.bestDetection,
    required this.debugImageFile,
  });
}

class WheelDetectorService {
  final String modelAsset;
  final int inputSize;
  final double confidenceThreshold;
  final double nmsThreshold;

  Interpreter? _interpreter;
  List<int>? _inputShape;
  List<int>? _outputShape;
  Future<void>? _loadFuture;

  WheelDetectorService({
    this.modelAsset = 'assets/models/best_float16.tflite',
    this.inputSize = 640,
    this.confidenceThreshold = 0.25,
    this.nmsThreshold = 0.45,
  });

  Future<void> _ensureLoaded() {
    _loadFuture ??= _load();
    return _loadFuture!;
  }

  Future<void> _load() async {
    _interpreter = await Interpreter.fromAsset(modelAsset);
    _inputShape = _interpreter!.getInputTensor(0).shape;
    _outputShape = _interpreter!.getOutputTensor(0).shape;

    debugPrint('[WheelDetector] Input tensor shape: $_inputShape');
    debugPrint('[WheelDetector] Output tensor shape: $_outputShape');
  }

  Future<WheelDetectionRunResult> detectBest(
    File imageFile, {
    bool saveDebugImage = true,
  }) async {
    await _ensureLoaded();

    final interpreter = _interpreter!;
    final decoded = img.decodeImage(await imageFile.readAsBytes());
    if (decoded == null) {
      throw Exception('Failed to decode image for wheel detection.');
    }

    final oriented = img.bakeOrientation(decoded);
    final prep = _letterbox(oriented, inputSize);

    final input = Float32List(inputSize * inputSize * 3);
    var offset = 0;
    for (var y = 0; y < inputSize; y++) {
      for (var x = 0; x < inputSize; x++) {
        final px = prep.image.getPixel(x, y);
        // Float detector models typically expect normalized 0..1 RGB.
        input[offset++] = px.r / 255.0;
        input[offset++] = px.g / 255.0;
        input[offset++] = px.b / 255.0;
      }
    }

    final inputTensor = input.reshape([1, inputSize, inputSize, 3]);
    final outputShape = _outputShape!;
    if (outputShape.length != 3) {
      throw Exception('Unexpected detector output shape: $outputShape');
    }

    final output = _create3dOutput(outputShape);
    interpreter.run(inputTensor, output);

    final candidates = _parseCandidates(
      output: output,
      shape: outputShape,
      prep: prep,
      originalWidth: oriented.width,
      originalHeight: oriented.height,
    );

    debugPrint(
      '[WheelDetector] Raw candidates above threshold: ${candidates.length}',
    );

    final kept = _nms(candidates, nmsThreshold);
    final best = kept.isEmpty ? null : kept.first;

    if (best != null) {
      debugPrint(
        '[WheelDetector] Final box: '
        'left=${best.left.toStringAsFixed(1)}, '
        'top=${best.top.toStringAsFixed(1)}, '
        'right=${best.right.toStringAsFixed(1)}, '
        'bottom=${best.bottom.toStringAsFixed(1)}, '
        'score=${best.score.toStringAsFixed(3)}',
      );
    } else {
      debugPrint('[WheelDetector] No final detection selected after NMS.');
    }

    File? debugFile;
    if (saveDebugImage && best != null) {
      debugFile = await _saveDebugOverlay(oriented, best);
      debugPrint('[WheelDetector] Debug image saved: ${debugFile.path}');
    }

    return WheelDetectionRunResult(
      bestDetection: best,
      debugImageFile: debugFile,
    );
  }

  List<List<List<double>>> _create3dOutput(List<int> shape) {
    return List.generate(
      shape[0],
      (_) => List.generate(shape[1], (_) => List.filled(shape[2], 0.0)),
    );
  }

  List<DetectionResult> _parseCandidates({
    required List<List<List<double>>> output,
    required List<int> shape,
    required _LetterboxPrep prep,
    required int originalWidth,
    required int originalHeight,
  }) {
    final out0 = output[0];
    final candidates = <DetectionResult>[];

    // Supports [1,5,8400] and [1,8400,5].
    if (shape[1] == 5) {
      final count = shape[2];
      for (var i = 0; i < count; i++) {
        final cx = out0[0][i];
        final cy = out0[1][i];
        final w = out0[2][i];
        final h = out0[3][i];
        final score = out0[4][i];
        _tryAddCandidate(
          candidates,
          cx,
          cy,
          w,
          h,
          score,
          prep,
          originalWidth,
          originalHeight,
        );
      }
    } else if (shape[2] == 5) {
      final count = shape[1];
      for (var i = 0; i < count; i++) {
        final cx = out0[i][0];
        final cy = out0[i][1];
        final w = out0[i][2];
        final h = out0[i][3];
        final score = out0[i][4];
        _tryAddCandidate(
          candidates,
          cx,
          cy,
          w,
          h,
          score,
          prep,
          originalWidth,
          originalHeight,
        );
      }
    } else {
      throw Exception('Unsupported detector output shape: $shape');
    }

    candidates.sort((a, b) => b.score.compareTo(a.score));
    return candidates;
  }

  void _tryAddCandidate(
    List<DetectionResult> out,
    double cx,
    double cy,
    double w,
    double h,
    double score,
    _LetterboxPrep prep,
    int originalWidth,
    int originalHeight,
  ) {
    if (!score.isFinite || score < confidenceThreshold) {
      return;
    }

    // Some exports return normalized values [0..1], some return 640-space values.
    final normalized =
        (cx.abs() <= 1.5 &&
        cy.abs() <= 1.5 &&
        w.abs() <= 1.5 &&
        h.abs() <= 1.5);

    var boxCx = cx;
    var boxCy = cy;
    var boxW = w;
    var boxH = h;
    if (normalized) {
      boxCx *= inputSize;
      boxCy *= inputSize;
      boxW *= inputSize;
      boxH *= inputSize;
    }

    final x1Model = boxCx - boxW / 2.0;
    final y1Model = boxCy - boxH / 2.0;
    final x2Model = boxCx + boxW / 2.0;
    final y2Model = boxCy + boxH / 2.0;

    // Reverse letterbox transform back to original-image coordinates.
    var left = (x1Model - prep.padX) / prep.scale;
    var top = (y1Model - prep.padY) / prep.scale;
    var right = (x2Model - prep.padX) / prep.scale;
    var bottom = (y2Model - prep.padY) / prep.scale;

    left = left.clamp(0.0, originalWidth.toDouble());
    top = top.clamp(0.0, originalHeight.toDouble());
    right = right.clamp(0.0, originalWidth.toDouble());
    bottom = bottom.clamp(0.0, originalHeight.toDouble());

    if (right - left < 2 || bottom - top < 2) {
      return;
    }

    out.add(
      DetectionResult(
        left: left,
        top: top,
        right: right,
        bottom: bottom,
        score: score,
      ),
    );
  }

  List<DetectionResult> _nms(List<DetectionResult> boxes, double iouThreshold) {
    if (boxes.isEmpty) return const [];

    final sorted = List<DetectionResult>.from(boxes)
      ..sort((a, b) => b.score.compareTo(a.score));

    final selected = <DetectionResult>[];
    while (sorted.isNotEmpty) {
      final current = sorted.removeAt(0);
      selected.add(current);
      sorted.removeWhere((b) => _iou(current, b) > iouThreshold);
    }

    return selected;
  }

  double _iou(DetectionResult a, DetectionResult b) {
    final interLeft = math.max(a.left, b.left);
    final interTop = math.max(a.top, b.top);
    final interRight = math.min(a.right, b.right);
    final interBottom = math.min(a.bottom, b.bottom);

    final interW = math.max(0.0, interRight - interLeft);
    final interH = math.max(0.0, interBottom - interTop);
    final interArea = interW * interH;

    final areaA = math.max(0.0, a.width) * math.max(0.0, a.height);
    final areaB = math.max(0.0, b.width) * math.max(0.0, b.height);
    final union = areaA + areaB - interArea;
    if (union <= 0) return 0.0;

    return interArea / union;
  }

  _LetterboxPrep _letterbox(img.Image source, int size) {
    final srcW = source.width.toDouble();
    final srcH = source.height.toDouble();
    final scale = math.min(size / srcW, size / srcH);

    final resizedW = (srcW * scale).round();
    final resizedH = (srcH * scale).round();
    final resized = img.copyResize(source, width: resizedW, height: resizedH);

    final canvas = img.Image(width: size, height: size);
    img.fill(canvas, color: img.ColorRgb8(114, 114, 114));

    final padX = (size - resizedW) / 2.0;
    final padY = (size - resizedH) / 2.0;

    img.compositeImage(canvas, resized, dstX: padX.round(), dstY: padY.round());

    return _LetterboxPrep(image: canvas, scale: scale, padX: padX, padY: padY);
  }

  Future<File> _saveDebugOverlay(img.Image image, DetectionResult box) async {
    final debug = img.Image.from(image);
    img.drawRect(
      debug,
      x1: box.left.round(),
      y1: box.top.round(),
      x2: box.right.round(),
      y2: box.bottom.round(),
      color: img.ColorRgb8(0, 255, 0),
      thickness: 3,
    );

    final bytes = img.encodeJpg(debug, quality: 92);
    final out = File(
      '${Directory.systemTemp.path}${Platform.pathSeparator}wheel_detection_debug_${DateTime.now().millisecondsSinceEpoch}.jpg',
    );
    await out.writeAsBytes(bytes, flush: true);
    return out;
  }

  void dispose() {
    _interpreter?.close();
  }
}

class _LetterboxPrep {
  final img.Image image;
  final double scale;
  final double padX;
  final double padY;

  const _LetterboxPrep({
    required this.image,
    required this.scale,
    required this.padX,
    required this.padY,
  });
}
