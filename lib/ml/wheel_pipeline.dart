import 'dart:io';
import 'prediction.dart';

abstract class WheelPipeline {
  Future<List<Prediction>> predict(File image);
}

class StubWheelPipeline implements WheelPipeline {
  @override
  Future<List<Prediction>> predict(File image) async {
    return [
      Prediction('Skoda_Style_01', 0.72),
      Prediction('Skoda_Style_14', 0.11),
      Prediction('Skoda_Style_07', 0.06),
      Prediction('Skoda_Style_03', 0.05),
      Prediction('Skoda_Style_19', 0.03),
    ];
  }
}