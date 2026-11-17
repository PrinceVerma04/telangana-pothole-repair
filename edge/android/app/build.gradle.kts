plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "app.gunthameter.edge"
    compileSdk = 34

    defaultConfig {
        applicationId = "app.gunthameter.edge"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
        // API base URL of the RoadFix server; override with -PgmApi=https://... at build time.
        buildConfigField("String", "GM_API", "\"${project.findProperty("gmApi") ?: "http://10.0.2.2:8000"}\"")
        buildConfigField("String", "CALIB_ID", "\"${project.findProperty("gmCalib") ?: "bus_default"}\"")
    }

    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.camera:camera-core:1.3.4")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")
    implementation("androidx.work:work-runtime-ktx:2.9.1")
    implementation("com.google.android.gms:play-services-location:21.3.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    // On-device segmenter (YOLO26n-seg exported by ml/export.py --formats litert16):
    // implementation("com.google.ai.edge.litert:litert:1.0.1")
}
