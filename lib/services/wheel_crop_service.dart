import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;

import '../models/detection_result.dart';

class WheelCropService {
  final double paddingRatio;
  final double tightenRatio;
  final bool enforceSquare;

  const WheelCropService({
    this.paddingRatio = 0.08,
    this.tightenRatio = 1.0,
    this.enforceSquare = true,
  });

  Future<File> cropDetectedWheel(
    File imageFile,
    DetectionResult detection, {
    String? outputPath,
  }) async {
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

    // Ispleciam i kvadrata apie centra, kad modelis matytu pati ratlanki, ne fona
    final cx = (left + right) / 2.0;
    final cy = (top + bottom) / 2.0;
    var side = enforceSquare ? math.max(width, height) : width;
    var sideY = enforceSquare ? side : height;

    // Po detekcijos leidziam kvadrata sugrieztinti, kad maziau foninio triuksmo patektu i embeddinga
    if (tightenRatio > 0 && tightenRatio.isFinite) {
      side *= tightenRatio;
      sideY *= tightenRatio;
    }
    if (side < 1.0) {
      side = 1.0;
    }
    if (sideY < 1.0) {
      sideY = 1.0;
    }

    if (enforceSquare) {
      // Saugiklis, kad kvadratas neisliptu uz nuotraukos ribu
      side = math.min(
        side,
        math.min(oriented.width.toDouble(), oriented.height.toDouble()),
      );
      sideY = side;
    } else {
      side = math.min(side, oriented.width.toDouble());
      sideY = math.min(sideY, oriented.height.toDouble());
    }

    var cropLeft = cx - side / 2.0;
    var cropTop = cy - sideY / 2.0;

    if (cropLeft < 0) cropLeft = 0;
    if (cropTop < 0) cropTop = 0;
    if (cropLeft + side > oriented.width) {
      cropLeft = oriented.width - side;
    }
    if (cropTop + sideY > oriented.height) {
      cropTop = oriented.height - sideY;
    }

    final x = cropLeft.round().clamp(0, oriented.width - 1).toInt();
    final y = cropTop.round().clamp(0, oriented.height - 1).toInt();
    final sW = side.round().clamp(1, oriented.width - x).toInt();
    final sH = sideY.round().clamp(1, oriented.height - y).toInt();

    final cropped = img.copyCrop(oriented, x: x, y: y, width: sW, height: sH);

    final out = outputPath == null
        ? File(
            '${Directory.systemTemp.path}${Platform.pathSeparator}wheel_crop_${DateTime.now().millisecondsSinceEpoch}.jpg',
          )
        : File(outputPath);
    await out.writeAsBytes(img.encodeJpg(cropped, quality: 94), flush: true);
    return out;
  }
}
