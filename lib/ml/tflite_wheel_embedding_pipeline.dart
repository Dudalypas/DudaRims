import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../core/constants/app_constants.dart';
import 'retrieval_reference.dart';
import 'retrieval_result.dart';

class TfliteWheelEmbeddingPipeline {
  final String modelAsset;
  final String referenceEmbeddingsAsset;
  final RetrievalReferenceMode referenceMode;
  final ClassAggregationMode aggregationMode;
  final int aggregationTopN;

  Interpreter? _interpreter;
  List<int>? _inputShape;
  List<int>? _outputShape;
  List<RetrievalReference>? _references;
  Future<void>? _loadFuture;

  TfliteWheelEmbeddingPipeline({
    required this.modelAsset,
    required this.referenceEmbeddingsAsset,
    this.referenceMode = RetrievalReferenceMode.multiReference,
    this.aggregationMode = ClassAggregationMode.topNSimilarityAverage,
    this.aggregationTopN = 3,
  });

  Future<void> _ensureLoaded() {
    _loadFuture ??= _load();
    return _loadFuture!;
  }

  Future<void> _load() async {
    _interpreter = await Interpreter.fromAsset(modelAsset);
    _inputShape = _interpreter!.getInputTensor(0).shape;
    _outputShape = _interpreter!.getOutputTensor(0).shape;

    final raw = await rootBundle.loadString(referenceEmbeddingsAsset);
    final decoded = jsonDecode(raw);
    if (decoded is! Map<String, dynamic>) {
      throw const FormatException('Invalid reference embeddings JSON.');
    }

    final classes = decoded['classes'];
    if (classes is! List) {
      throw const FormatException(
        'Reference embeddings JSON has no classes list.',
      );
    }

    final refs = <RetrievalReference>[];
    for (final item in classes) {
      if (item is! Map<String, dynamic>) continue;
      final label = (item['label'] ?? '').toString().trim();
      final refsJson = item['references'];
      final embeddings = <List<double>>[];
      if (refsJson is List) {
        for (final ref in refsJson) {
          if (ref is! Map<String, dynamic>) continue;
          final embJson = ref['embedding'];
          if (embJson is! List) continue;
          final emb = <double>[];
          for (final v in embJson) {
            if (v is num) {
              emb.add(v.toDouble());
            }
          }
          if (emb.isNotEmpty) {
            embeddings.add(_l2Normalize(emb));
          }
        }
      }

      // Suderinamumas su senesniu centroid-only JSON: classes[].embedding
      if (embeddings.isEmpty) {
        final legacyEmbeddingJson = item['embedding'];
        if (legacyEmbeddingJson is List) {
          final legacyEmbedding = <double>[];
          for (final v in legacyEmbeddingJson) {
            if (v is num) {
              legacyEmbedding.add(v.toDouble());
            }
          }
          if (legacyEmbedding.isNotEmpty) {
            embeddings.add(_l2Normalize(legacyEmbedding));
          }
        }
      }

      if (label.isEmpty || embeddings.isEmpty) continue;

      List<double>? centroidEmbedding;
      final centroidJson = item['centroid_embedding'];
      if (centroidJson is List) {
        final c = <double>[];
        for (final v in centroidJson) {
          if (v is num) {
            c.add(v.toDouble());
          }
        }
        if (c.isNotEmpty) {
          centroidEmbedding = _l2Normalize(c);
        }
      }

      centroidEmbedding ??= _meanEmbedding(embeddings);

      refs.add(
        RetrievalReference(
          label: label,
          embeddings: embeddings,
          centroidEmbedding: centroidEmbedding,
          sampleCount: (item['sample_count'] is num)
              ? (item['sample_count'] as num).toInt()
              : 0,
        ),
      );
    }

    if (refs.isEmpty) {
      throw const FormatException('No valid reference embeddings found.');
    }

    _references = refs;

    if (AppConstants.enablePipelineDebugLogs) {
      debugPrint(
        '[pipeline] model=$modelAsset input=$_inputShape output=$_outputShape '
        'refs=${refs.length} mode=$referenceMode agg=$aggregationMode topN=$aggregationTopN',
      );
    }
  }

  Future<List<double>> embed(File imageFile) async {
    await _ensureLoaded();

    final interpreter = _interpreter!;
    final inputShape = _inputShape!;
    final outputShape = _outputShape!;

    if (inputShape.length != 4) {
      throw Exception('Unexpected embedding model input shape: $inputShape');
    }

    final h = inputShape[1];
    final w = inputShape[2];

    final decoded = img.decodeImage(await imageFile.readAsBytes());
    if (decoded == null) {
      throw Exception('Failed to decode image for embedding extraction.');
    }

    final oriented = img.bakeOrientation(decoded);
    final resized = img.copyResize(oriented, width: w, height: h);

    final input = Float32List(1 * h * w * 3);
    var i = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final px = resized.getPixel(x, y);
        // RGB float32 lieka [0..255], kaip tikisi eksportuotas modelis
        input[i++] = px.r.toDouble();
        input[i++] = px.g.toDouble();
        input[i++] = px.b.toDouble();
      }
    }

    final inputTensor = input.reshape([1, h, w, 3]);

    if (outputShape.isEmpty) {
      throw const FormatException('Embedding model output shape is empty.');
    }

    final outputSize = outputShape.skip(1).fold<int>(1, (a, b) => a * b);
    final output = List.generate(1, (_) => List.filled(outputSize, 0.0));

    interpreter.run(inputTensor, output);
    final embedding = _l2Normalize(output[0]);
    if (AppConstants.enablePipelineDebugLogs) {
      debugPrint(
        '[pipeline] embedding done size=${embedding.length}',
      );
    }
    return embedding;
  }

  Future<List<RetrievalResult>> retrieveTopK(
    File imageFile, {
    int k = 5,
  }) async {
    final query = await embed(imageFile);
    final refs = _references!;

    final scored = <RetrievalResult>[];
    for (final ref in refs) {
      final similarity = _classSimilarity(query, ref);
      scored.add(
        RetrievalResult(label: ref.label, cosineSimilarity: similarity),
      );
    }
    scored.sort((a, b) => b.cosineSimilarity.compareTo(a.cosineSimilarity));

    return scored.take(math.max(1, k)).toList(growable: false);
  }

  double _classSimilarity(List<double> query, RetrievalReference ref) {
    if (referenceMode == RetrievalReferenceMode.centroidBaseline) {
      final centroid = ref.centroidEmbedding;
      if (centroid == null || centroid.isEmpty) {
        return 0.0;
      }
      return _cosineSimilarity(query, centroid);
    }

    if (ref.embeddings.isEmpty) {
      return 0.0;
    }

    final sims = ref.embeddings
        .map((e) => _cosineSimilarity(query, e))
        .toList();
    if (sims.isEmpty) {
      return 0.0;
    }

    sims.sort((a, b) => b.compareTo(a));

    switch (aggregationMode) {
      case ClassAggregationMode.maxSimilarity:
        return sims.first;
      case ClassAggregationMode.topNSimilarityAverage:
        final topN = math.max(1, math.min(aggregationTopN, sims.length));
        final selected = sims.take(topN);
        final sum = selected.fold<double>(0.0, (acc, v) => acc + v);
        return sum / topN;
    }
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    final n = math.min(a.length, b.length);
    if (n == 0) return 0.0;

    var dot = 0.0;
    for (var i = 0; i < n; i++) {
      dot += a[i] * b[i];
    }
    return dot;
  }

  List<double> _l2Normalize(List<double> v) {
    var sum = 0.0;
    for (final x in v) {
      sum += x * x;
    }
    final norm = math.sqrt(sum);
    if (!norm.isFinite || norm <= 1e-12) {
      return List<double>.filled(v.length, 0.0);
    }
    return v.map((x) => x / norm).toList(growable: false);
  }

  List<double> _meanEmbedding(List<List<double>> vectors) {
    if (vectors.isEmpty) {
      return const [];
    }
    final length = vectors.first.length;
    final acc = List<double>.filled(length, 0.0);
    var count = 0;

    for (final v in vectors) {
      if (v.length != length) continue;
      count += 1;
      for (var i = 0; i < length; i++) {
        acc[i] += v[i];
      }
    }

    if (count == 0) {
      return const [];
    }

    for (var i = 0; i < length; i++) {
      acc[i] /= count;
    }
    return _l2Normalize(acc);
  }

  void dispose() {
    _interpreter?.close();
  }
}
