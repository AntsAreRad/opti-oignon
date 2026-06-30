#include "veilid_bridge.h"

/*
 * SKELETON stubs. Every entry returns a not-implemented sentinel:
 *   - jlong / jint returns: -1
 *   - jbyteArray / jstring returns: nullptr
 *
 * On the Kotlin side, a null from appCall is read as "no reply" (Bulbe / route
 * down), and a null from a route/record call is read as "unavailable". A -1 int
 * is a generic failure. Host-side, replace each body with a call into
 * veilid-core and marshal the result.
 */

#define OO_NOT_IMPLEMENTED (-1)

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_optioignon_veilid_VeilidBridge_nodeInit(JNIEnv *, jobject, jstring) {
    return OO_NOT_IMPLEMENTED;
}

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_nodeAttach(JNIEnv *, jobject, jlong) {
    return OO_NOT_IMPLEMENTED;
}

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_nodeDetach(JNIEnv *, jobject, jlong) {
    return OO_NOT_IMPLEMENTED;
}

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_nodeShutdown(JNIEnv *, jobject, jlong) {
    return OO_NOT_IMPLEMENTED;
}

JNIEXPORT jbyteArray JNICALL
Java_org_optioignon_veilid_VeilidBridge_routeAllocate(JNIEnv *, jobject, jlong) {
    return nullptr;
}

JNIEXPORT jstring JNICALL
Java_org_optioignon_veilid_VeilidBridge_routeImport(JNIEnv *, jobject, jlong, jbyteArray) {
    return nullptr;
}

JNIEXPORT jbyteArray JNICALL
Java_org_optioignon_veilid_VeilidBridge_appCall(JNIEnv *, jobject, jlong, jstring, jbyteArray) {
    return nullptr;
}

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_recordOpen(JNIEnv *, jobject, jlong, jstring) {
    return OO_NOT_IMPLEMENTED;
}

JNIEXPORT jbyteArray JNICALL
Java_org_optioignon_veilid_VeilidBridge_recordGet(JNIEnv *, jobject, jlong, jstring, jint) {
    return nullptr;
}

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_recordSet(JNIEnv *, jobject, jlong, jstring, jint, jbyteArray) {
    return OO_NOT_IMPLEMENTED;
}

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_recordClose(JNIEnv *, jobject, jlong, jstring) {
    return OO_NOT_IMPLEMENTED;
}

}  // extern "C"
