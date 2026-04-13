import 'rim_variant.dart';

class RimModel {
  final String className;
  final String displayName;
  final List<RimVariant> variants;

  const RimModel({
    required this.className,
    required this.displayName,
    required this.variants,
  });
}
