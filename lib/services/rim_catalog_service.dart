import '../models/rim_model.dart';
import '../models/rim_variant.dart';
import '../models/wheel_spec.dart';

class RimCatalogService {
  RimModel? resolveByClass(String predictedClass, List<WheelSpec> specs) {
    final target = predictedClass.trim().toLowerCase();
    if (target.isEmpty) return null;

    final matches = specs.where((w) => w.wheelClassName.trim().toLowerCase() == target).toList();
    if (matches.isEmpty) {
      return null;
    }

    final variants = <RimVariant>[];
    final seen = <String>{};
    for (final spec in matches) {
      final label = _variantLabel(spec);
      if (seen.add(label)) {
        variants.add(RimVariant(variantLabel: label, source: spec));
      }
    }

    variants.sort((a, b) => a.variantLabel.compareTo(b.variantLabel));

    return RimModel(
      className: predictedClass,
      displayName: matches.first.officialWheelName ?? predictedClass,
      variants: variants,
    );
  }

  String _variantLabel(WheelSpec spec) {
    final d = spec.diameterIn;
    final width = spec.widthJ;
    if (d == null) {
      return spec.officialWheelName ?? 'Variantas';
    }

    final diameterText = d % 1 == 0 ? '${d.toInt()}"' : '${d.toStringAsFixed(1)}"';
    if (width == null) return diameterText;
    final widthText = width % 1 == 0 ? width.toInt().toString() : width.toStringAsFixed(1);
    return '$diameterText  •  ${widthText}J';
  }
}
