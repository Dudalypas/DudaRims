import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../../core/components/action_button_styles.dart';
import '../../core/constants/app_constants.dart';
import '../../core/state/app_state.dart';
import '../../core/theme/theme_tokens.dart';
import '../../services/embedding_retrieval_recognition_service.dart';
import '../../services/local_fitment_repository.dart';
import '../../services/mock_recognition_service.dart';
import '../../services/recognition_service.dart';
import '../car/car_selection_screen.dart';
import '../result/recognition_result_screen.dart';
import '../settings/settings_screen.dart';

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  final _picker = ImagePicker();
  late final RecognitionService _recognitionService;

  XFile? _preview;
  bool _isProcessing = false;
  String? _processingError;

  void _clearPreview() {
    if (!mounted) return;
    setState(() {
      _preview = null;
      _processingError = null;
      _isProcessing = false;
    });
  }

  @override
  void initState() {
    super.initState();
    final repository = LocalFitmentRepository();

    if (!AppConstants.enableRealMlInference) {
      _recognitionService = MockRecognitionService(repository);
      if (AppConstants.enablePipelineDebugLogs) {
        debugPrint('[recognition] capture init: mock');
      }
      return;
    }

    _recognitionService = EmbeddingRetrievalRecognitionService(
      repository: repository,
    );

    if (AppConstants.enablePipelineDebugLogs) {
      debugPrint('[recognition] capture init: ${_recognitionService.runtimeType}');
    }
  }

  Future<void> _pick(ImageSource source) async {
    final img = await _picker.pickImage(
      source: source,
      imageQuality: 92,
      maxWidth: 1600,
    );
    if (img == null) return;
    if (!mounted) return;

    final imageFile = File(img.path);
    setState(() {
      _preview = img;
      _isProcessing = true;
      _processingError = null;
    });

    var keepTrying = true;
    while (keepTrying) {
      final stopwatch = Stopwatch()..start();
      try {
        if (AppConstants.enablePipelineDebugLogs) {
          debugPrint('[recognition] analyze start: ${_recognitionService.runtimeType}');
        }

        final outcomeFuture = _recognitionService.analyze(imageFile);
        final outcome = await outcomeFuture;
        stopwatch.stop();
        debugPrint('[perf] recognition_total_ms=${stopwatch.elapsedMilliseconds}');

        if (!mounted) return;
        final action = await Navigator.of(context)
            .push<RecognitionResultAction>(
              MaterialPageRoute(
                builder: (_) => RecognitionResultScreen(
                  imageFile: imageFile,
                  outcome: outcome,
                ),
              ),
            );

        if (action == RecognitionResultAction.retrySamePhoto) {
          if (!mounted) return;
          setState(() {
            _isProcessing = true;
            _processingError = null;
          });
          continue;
        }

        keepTrying = false;
      } catch (e) {
        stopwatch.stop();
        debugPrint('[perf] recognition_failed_ms=${stopwatch.elapsedMilliseconds}');
        if (!mounted) return;
        setState(() {
          _processingError = 'Nepavyko atpažinti ratlankio: $e';
        });
        keepTrying = false;
      }
    }

    _clearPreview();
  }

  @override
  Widget build(BuildContext context) {
    final selectedCar = context.watch<AppState>().selectedCar;
    final hasPreview = _preview != null;
    final carLabel = selectedCar == null
        ? 'Pasirinkti automobilį'
        : '${selectedCar.model ?? 'Skoda'} ${selectedCar.generationLabel}';
    final colors = Theme.of(context).colorScheme;

    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 280),
              child: hasPreview
                  ? Image.file(
                      File(_preview!.path),
                      fit: BoxFit.cover,
                      key: ValueKey(_preview!.path),
                    )
                  : Container(
                      key: const ValueKey('camera-bg'),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            colors.surfaceContainerHighest,
                            colors.surfaceContainerLow,
                          ],
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                        ),
                      ),
                    ),
            ),
          ),
          Positioned.fill(
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    colors.cameraOverlaySoft,
                    Colors.transparent,
                    colors.cameraOverlayStrong,
                  ],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
            ),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: ActionChip(
                          avatar: const Icon(
                            Icons.directions_car_filled_outlined,
                          ),
                          label: Text(carLabel),
                          onPressed: () {
                            Navigator.of(context).push(
                              MaterialPageRoute(
                                builder: (_) => const CarSelectionScreen(),
                              ),
                            );
                          },
                        ),
                      ),
                      const SizedBox(width: 10),
                      IconButton.filledTonal(
                        onPressed: () {
                          Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (_) => const SettingsScreen(),
                            ),
                          );
                        },
                        icon: const Icon(Icons.settings_outlined),
                        tooltip: 'Nustatymai',
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  Text(
                    'Identifikuokite ratlankį',
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Nufotografuokite arba įkelkite ratlankio nuotrauką. Tai Identifikavimo žingsnis, o suderinamumas su automobiliu bus tikrinamas po to.',
                    style: TextStyle(color: colors.mutedText),
                  ),
                  const SizedBox(height: 18),
                  AnimatedSwitcher(
                    duration: const Duration(milliseconds: 220),
                    child: hasPreview
                        ? ClipRRect(
                            key: ValueKey(_preview!.path),
                            borderRadius: BorderRadius.circular(18),
                            child: AspectRatio(
                              aspectRatio: 4 / 3,
                              child: Image.file(
                                File(_preview!.path),
                                fit: BoxFit.cover,
                              ),
                            ),
                          )
                        : Container(
                            key: const ValueKey('hub-placeholder'),
                            width: double.infinity,
                            padding: const EdgeInsets.all(20),
                            decoration: BoxDecoration(
                              color: colors.panelSurface,
                              borderRadius: BorderRadius.circular(18),
                              border: Border.all(color: colors.outlineVariant),
                            ),
                            child: Row(
                              children: [
                                Icon(
                                  Icons.photo_camera_back_outlined,
                                  color: colors.primary,
                                  size: 28,
                                ),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Text(
                                    'Pasirinkite įvesties būdą ir pradėkite ratlankio identifikavimą.',
                                    style: TextStyle(color: colors.mutedText),
                                  ),
                                ),
                              ],
                            ),
                          ),
                  ),
                  const SizedBox(height: 20),
                  if (_isProcessing)
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(14),
                      decoration: BoxDecoration(
                        color: colors.panelSurface,
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Apdorojama...',
                            style: TextStyle(
                              fontWeight: FontWeight.w700,
                              color: colors.onSurface,
                            ),
                          ),
                          const SizedBox(height: 10),
                          const LinearProgressIndicator(minHeight: 4),
                        ],
                      ),
                    ),
                  if (_processingError != null) ...[
                    const SizedBox(height: 12),
                    Text(
                      _processingError!,
                      style: TextStyle(color: colors.error),
                    ),
                  ],
                  const Spacer(),
                  ActionButtonStyles.primaryButton(
                    context: context,
                    onPressed: _isProcessing
                        ? null
                        : () => _pick(ImageSource.camera),
                    label: 'Fotografuoti',
                    icon: const Icon(Icons.camera_alt_rounded),
                  ),
                  const SizedBox(height: ActionButtonStyles.buttonStackSpacing),
                  ActionButtonStyles.secondaryButton(
                    context: context,
                    onPressed: _isProcessing
                        ? null
                        : () => _pick(ImageSource.gallery),
                    label: 'Įkelti nuotrauką',
                    icon: const Icon(Icons.photo_library_outlined),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
