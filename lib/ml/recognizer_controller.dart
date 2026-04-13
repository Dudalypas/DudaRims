import 'dart:io';
import 'package:flutter/foundation.dart';
import 'prediction.dart';
import 'wheel_pipeline.dart';

class RecognizerState {
  final bool isLoading;
  final String? error;
  final List<Prediction> results;

  const RecognizerState({
    this.isLoading = false,
    this.error,
    this.results = const [],
  });

  RecognizerState copyWith({
    bool? isLoading,
    String? error,
    List<Prediction>? results,
  }) {
    return RecognizerState(
      isLoading: isLoading ?? this.isLoading,
      error: error,
      results: results ?? this.results,
    );
  }
}

class RecognizerController extends ValueNotifier<RecognizerState> {
  final WheelPipeline pipeline;

  RecognizerController(this.pipeline) : super(const RecognizerState());

  Future<void> recognize(File image) async {
    value = value.copyWith(isLoading: true, error: null, results: const []);
    try {
      final res = await pipeline.predict(image);
      value = value.copyWith(isLoading: false, results: res);
    } catch (e) {
      value = value.copyWith(isLoading: false, error: e.toString());
    }
  }

  void reset() {
    value = const RecognizerState();
  }
}