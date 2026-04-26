import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

class FitmentDatabase {
  static const _assetDbPath = 'assets/data/fitment.sqlite3';
  static const _dbFileName = 'fitment.sqlite3';

  static final FitmentDatabase instance = FitmentDatabase._();

  Database? _db;

  FitmentDatabase._();

  Future<void> prewarm() async {
    await open();
  }

  Future<Database> open() async {
    final existing = _db;
    if (existing != null) return existing;

    final dbDir = await getDatabasesPath();
    final dbPath = p.join(dbDir, _dbFileName);
    await _copyAssetDbIfMissing(dbPath);

    final db = await openDatabase(dbPath, readOnly: true);
    _db = db;
    return db;
  }

  Future<List<Map<String, Object?>>> loadCarModels() async {
    final db = await open();
    return db.query('car_models');
  }

  Future<List<Map<String, Object?>>> loadWheelModels() async {
    final db = await open();
    return db.query('wheel_models');
  }

  Future<void> _copyAssetDbIfMissing(String targetPath) async {
    final file = File(targetPath);
    if (await file.exists()) return;

    await file.parent.create(recursive: true);

    final data = await rootBundle.load(_assetDbPath);
    final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await file.writeAsBytes(bytes, flush: true);
  }
}
