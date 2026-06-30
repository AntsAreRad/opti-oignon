// Opti-Oignon mobile client (SKELETON). Validated host-side; see BUILD_RUNBOOK.md.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "opti-oignon-mobile"
include(":app", ":veilid-bridge")
