import 'package:flutter/material.dart';
import 'app/app.dart';
import 'app/state/app_state.dart';

Future<void> main() async {
	WidgetsFlutterBinding.ensureInitialized();
	final appState = await AppState.create();
	runApp(App(appState: appState));
}