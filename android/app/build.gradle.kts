plugins {
    id("com.android.application")
    id("kotlin-android")
    // Flutter gradle pluginas turi eiti po android pluginu, kitaip konfigas suluzta
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.dudarims"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Pries publish pasikeisti i savo unikalų applicationId.
        applicationId = "com.example.dudarims"
        // Min/target/version pasiimti is Flutter configo, kad neissiskirtu tarp platformu
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // TODO: Realiam release reikia atskiro signing configo, paziureti pries publish irgi
            // Kol kas cia debug key, kad veiktu lokalus `flutter run --release` testas
            // Pasicheckinti viska pries publish, nes gali buti dar reikalavimu
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}
