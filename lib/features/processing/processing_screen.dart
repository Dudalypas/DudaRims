import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';

import '../../core/constants/app_constants.dart';
import '../../services/embedding_retrieval_recognition_service.dart';
import '../../services/local_fitment_repository.dart';
import '../../services/mock_recognition_service.dart';
import '../../services/recognition_service.dart';
import '../../services/tflite_recognition_service.dart';
import '../result/recognition_result_screen.dart';

class ProcessingScreen extends StatefulWidget {
  final File imageFile;

  const ProcessingScreen({super.key, required this.imageFile});

  @override
  State<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen> {
  late final RecognitionService _service;

  int _messageIndex = 0;
  Timer? _timer;
  String? _error;

  static const _messages = [
    'Analizuojama ratlankio geometrija...',
    'Lyginama su modelių duomenų baze...',
    'Paruošiamas rezultatas...',
  ];

  @override
  void initState() {
    super.initState();
    final repository = LocalFitmentRepository();
    if (!AppConstants.enableRealMlInference) {
      _service = MockRecognitionService(repository);
      if (AppConstants.enablePipelineDebugLogs) {
        debugPrint('[RecognitionRouting] Using mock recognition service.');
      }
    } else {
      final classifierBaseline = TfliteRecognitionService(
        repository: repository,
      );
      _service = switch (AppConstants.recognitionPipelineMode) {
        RecognitionPipelineMode.embeddingRetrieval =>
          EmbeddingRetrievalRecognitionService(
            repository: repository,
            fallbackService: classifierBaseline,
          ),
        RecognitionPipelineMode.classifierBaseline => classifierBaseline,
      };
      if (AppConstants.enablePipelineDebugLogs) {
        debugPrint(
          '[RecognitionRouting] mode=${AppConstants.recognitionPipelineMode} '
          'fallbackEnabled=${AppConstants.enableClassifierFallback}',
        );
      }
    }
    _timer = Timer.periodic(const Duration(milliseconds: 900), (_) {
      if (!mounted) return;
      setState(() {
        _messageIndex = (_messageIndex + 1) % _messages.length;
      });
    });
    _run();
  }

  Future<void> _run() async {
    try {
      final outcome = await _service.analyze(widget.imageFile);
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => RecognitionResultScreen(
            imageFile: widget.imageFile,
            outcome: outcome,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Nepavyko atlikti analizės: $e';
      });
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 92,
                height: 92,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    colors: [colors.primary, colors.tertiary],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                child: const Icon(Icons.radar_rounded, size: 40),
              ),
              const SizedBox(height: 20),
              const SizedBox(
                width: 180,
                child: LinearProgressIndicator(minHeight: 5),
              ),
              const SizedBox(height: 16),
              Text(
                _error ?? _messages[_messageIndex],
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              if (_error != null) ...[
                const SizedBox(height: 16),
                FilledButton(
                  onPressed: () {
                    setState(() => _error = null);
                    _run();
                  },
                  child: const Text('Bandyti dar kartą'),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
