#ifndef OPTIOIGNON_VEILID_BRIDGE_H
#define OPTIOIGNON_VEILID_BRIDGE_H

/*
 * JNI bridge to veilid-core.
 *
 * SKELETON: prototypes only; veilid_bridge.cpp returns not-implemented
 * sentinels. The real implementation links veilid-core and is built host-side
 * (see ../../../BUILD_RUNBOOK.md). These signatures are the contract
 * VeilidBridge.kt binds to; the JNI names mangle the class
 * org/optioignon/veilid/VeilidBridge.
 */

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_org_optioignon_veilid_VeilidBridge_nodeInit(JNIEnv *, jobject, jstring);

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_nodeAttach(JNIEnv *, jobject, jlong);

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_nodeDetach(JNIEnv *, jobject, jlong);

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_nodeShutdown(JNIEnv *, jobject, jlong);

JNIEXPORT jbyteArray JNICALL
Java_org_optioignon_veilid_VeilidBridge_routeAllocate(JNIEnv *, jobject, jlong);

JNIEXPORT jstring JNICALL
Java_org_optioignon_veilid_VeilidBridge_routeImport(JNIEnv *, jobject, jlong, jbyteArray);

JNIEXPORT jbyteArray JNICALL
Java_org_optioignon_veilid_VeilidBridge_appCall(JNIEnv *, jobject, jlong, jstring, jbyteArray);

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_recordOpen(JNIEnv *, jobject, jlong, jstring);

JNIEXPORT jbyteArray JNICALL
Java_org_optioignon_veilid_VeilidBridge_recordGet(JNIEnv *, jobject, jlong, jstring, jint);

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_recordSet(JNIEnv *, jobject, jlong, jstring, jint, jbyteArray);

JNIEXPORT jint JNICALL
Java_org_optioignon_veilid_VeilidBridge_recordClose(JNIEnv *, jobject, jlong, jstring);

#ifdef __cplusplus
}
#endif

#endif  // OPTIOIGNON_VEILID_BRIDGE_H
