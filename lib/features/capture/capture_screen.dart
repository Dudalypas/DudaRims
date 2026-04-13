import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../../app/state/app_state.dart';
import '../../ml/recognizer_controller.dart';
import '../../ml/prediction.dart';
import '../../models/detection_result.dart';
import '../car/car_picker_sheet.dart';
import '../settings/settings_screen.dart';
import '../../services/wheel_crop_service.dart';
import '../../services/wheel_detector_service.dart';
import '../result/result_screen.dart';
import '../../ml/tflite_wheel_pipeline.dart';

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  final _picker = ImagePicker();
  XFile? _image;

  final _controller = RecognizerController(TfliteWheelPipeline());
  final _detector = WheelDetectorService();
  final _cropService = const WheelCropService();

  File? _classificationImage;
  File? _debugImage;
  DetectionResult? _detection;
  String _pipelineMessage = '';
  bool _isProcessing = false;
  String _processingText = 'Analizuojama...';

  Future<void> _openCarPicker() async {
    final appState = context.read<AppState>();
    final selected = await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (_) => CarPickerSheet(
        fitments: appState.vehicleFitments,
        initialSelection: appState.selectedCar,
      ),
    );

    if (!mounted) return;
    await appState.setSelectedCar(selected);
  }

  Future<void> _pick(ImageSource source) async {
    final img = await _picker.pickImage(
      source: source,
      imageQuality: 92,
      maxWidth: 1600,
    );
    if (img == null) return;
    setState(() => _image = img);
  }

  Future<void> _recognize() async {
    final img = _image;
    if (img == null) return;

    setState(() {
      _isProcessing = true;
      _processingText = 'Analizuojama ratlankio geometrija...';
    });

    final originalFile = File(img.path);
    File classifierInput = originalFile;
    DetectionResult? detection;
    File? debugImage;
    var message = 'Ratas neaptiktas, klasifikacijai naudota originali nuotrauka';

    try {
      final detectionRun = await _detector.detectBest(
        originalFile,
        saveDebugImage: true,
      );
      detection = detectionRun.bestDetection;
      debugImage = detectionRun.debugImageFile;

      if (detection != null) {
        classifierInput = await _cropService.cropDetectedWheel(originalFile, detection);
        message = 'Ratas aptiktas ir apkirptas prieš klasifikaciją';
      }

      if (mounted) {
        setState(() {
          _processingText = 'Lyginama su ratlankių duomenų baze...';
        });
      }
    } catch (e) {
      message = 'Rato detektorius nesuveikė, klasifikacijai naudota originali nuotrauka';
      debugPrint('[WheelDetector] Failure: $e');
    }

    setState(() {
      _classificationImage = classifierInput;
      _debugImage = debugImage;
      _detection = detection;
      _pipelineMessage = message;
    });

    // Hook point: replace this with another recognizer backend if model pipeline changes.
    await _controller.recognize(classifierInput);
    final List<Prediction> predictions = _controller.value.results;

    if (!mounted) return;
    setState(() {
      _isProcessing = false;
    });

    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ResultScreen(
          imageFile: originalFile,
          classifierInputImageFile: _classificationImage,
          detectorDebugImageFile: _debugImage,
          detection: _detection,
          pipelineMessage: _pipelineMessage,
          predictions: predictions,
        ),
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    _detector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final hasImage = _image != null;
    final appState = context.watch<AppState>();
    final selectedCar = appState.selectedCar;
    final carLabel = selectedCar == null
        ? 'Pasirink automobilį'
        : '${selectedCar.model} ${selectedCar.generationLabel}';

    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(
            child: !hasImage
                ? Container(
                    decoration: const BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [Color(0xFF101722), Color(0xFF0C1017)],
                      ),
                    ),
                    child: const Center(
                      child: Text('Nufotografuok arba pasirink ratlankio nuotrauką'),
                    ),
                  )
                : Image.file(File(_image!.path), fit: BoxFit.cover),
          ),
          Positioned.fill(
            child: DecoratedBox(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Colors.black.withValues(alpha: 0.55), Colors.black.withValues(alpha: 0.25), Colors.black.withValues(alpha: 0.72)],
                  stops: const [0.0, 0.45, 1.0],
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
                          avatar: const Icon(Icons.directions_car, size: 18),
                          label: Text(carLabel, overflow: TextOverflow.ellipsis),
                          onPressed: appState.isLoadingData ? null : _openCarPicker,
                        ),
                      ),
                      const SizedBox(width: 8),
                      IconButton.filledTonal(
                        onPressed: () {
                          Navigator.of(context).push(
                            MaterialPageRoute(builder: (_) => const SettingsScreen()),
                          );
                        },
                        icon: const Icon(Icons.tune),
                      ),
                    ],
                  ),
                  const Spacer(),
                  Container(
                    width: 280,
                    height: 280,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(color: Colors.white.withValues(alpha: 0.6), width: 1.2),
                    ),
                    child: Center(
                      child: Icon(Icons.radio_button_unchecked_rounded, size: 180, color: Colors.white.withValues(alpha: 0.22)),
                    ),
                  ),
                  const Spacer(),
                  Row(
                    children: [
                      IconButton.filledTonal(
                        onPressed: () => _pick(ImageSource.gallery),
                        icon: const Icon(Icons.photo_library_outlined),
                      ),
                      const Spacer(),
                      InkWell(
                        borderRadius: BorderRadius.circular(44),
                        onTap: hasImage ? _recognize : () => _pick(ImageSource.camera),
                        child: Container(
                          width: 88,
                          height: 88,
                          padding: const EdgeInsets.all(5),
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(color: Colors.white, width: 3),
                          ),
                          child: Container(
                            decoration: const BoxDecoration(shape: BoxShape.circle, color: Colors.white),
                          ),
                        ),
                      ),
                      const Spacer(),
                      IconButton.filledTonal(
                        onPressed: () => _pick(ImageSource.camera),
                        icon: const Icon(Icons.camera_alt_outlined),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  const Text('Nuskenuok ratlankį ir gauk suderinamumo įvertinimą', style: TextStyle(color: Colors.white70)),
                ],
              ),
            ),
          ),
          if (_isProcessing)
            Positioned.fill(
              child: Container(
                color: Colors.black.withValues(alpha: 0.66),
                child: Center(
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const SizedBox(
                            width: 44,
                            height: 44,
                            child: CircularProgressIndicator(strokeWidth: 3),
                          ),
                          const SizedBox(height: 14),
                          Text(_processingText, style: const TextStyle(fontWeight: FontWeight.w600)),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}