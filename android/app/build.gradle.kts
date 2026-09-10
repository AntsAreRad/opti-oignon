plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.serialization")
}

android {
    namespace = "org.optioignon.mobile"
    compileSdk = 34

    defaultConfig {
        applicationId = "org.optioignon.mobile"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0-skeleton"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation(project(":veilid-bridge"))
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.1")

    // The wire envelopes are decidable on a JVM alone, so they are checked in
    // src/test rather than src/androidTest: no device, no emulator.
    testImplementation("junit:junit:4.13.2")
}
