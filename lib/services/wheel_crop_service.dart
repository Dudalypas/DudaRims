import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;

import '../models/detection_result.dart';

class WheelCropService {
  final double paddingRatio;

  const WheelCropService({this.paddingRatio = 0.08});

  Future<File> cropDetectedWheel(
    File imageFile,
    DetectionResult detection,
  ) async {
    final decoded = img.decodeImage(await imageFile.readAsBytes());
    if (decoded == null) {
      throw Exception('Failed to decode image for wheel crop.');
    }

    final oriented = img.bakeOrientation(decoded);

    var left = detection.left;
    var top = detection.top;
    var right = detection.right;
    var bottom = detection.bottom;

    var width = right - left;
    var height = bottom - top;

    final padX = width * paddingRatio;
    final padY = height * paddingRatio;

    left -= padX;
    right += padX;
    top -= padY;
    bottom += padY;

    width = right - left;
    height = bottom - top;

    // Expand to square around the current center so classifier gets focused wheel crop.
    final cx = (left + right) / 2.0;
    final cy = (top + bottom) / 2.0;
    var side = math.max(width, height);

    // Clamp side to image bounds.
    side = math.min(
      side,
      math.min(oriented.width.toDouble(), oriented.height.toDouble()),
    );

    var cropLeft = cx - side / 2.0;
    var cropTop = cy - side / 2.0;

    if (cropLeft < 0) cropLeft = 0;
    if (cropTop < 0) cropTop = 0;
    if (cropLeft + side > oriented.width) {
      cropLeft = oriented.width - side;
    }
    if (cropTop + side > oriented.height) {
      cropTop = oriented.height - side;
    }

    final x = cropLeft.round().clamp(0, oriented.width - 1).toInt();
    final y = cropTop.round().clamp(0, oriented.height - 1).toInt();
    final s = side
        .round()
        .clamp(1, math.min(oriented.width - x, oriented.height - y))
        .toInt();

    final cropped = img.copyCrop(oriented, x: x, y: y, width: s, height: s);

    final out = File(
      '${Directory.systemTemp.path}${Platform.pathSeparator}wheel_crop_${DateTime.now().millisecondsSinceEpoch}.jpg',
    );
    await out.writeAsBytes(img.encodeJpg(cropped, quality: 94), flush: true);
    return out;
  }
}
