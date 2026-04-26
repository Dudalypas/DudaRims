import 'package:flutter/material.dart';

import 'app/app.dart';
import 'core/state/app_state.dart';
import 'services/fitment_database.dart';

Future<void> main() async {
	WidgetsFlutterBinding.ensureInitialized();
	await FitmentDatabase.instance.prewarm();
	final appState = AppState();
	await appState.initialize();
	runApp(App(appState: appState));
}