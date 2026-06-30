package org.optioignon.veilid

/**
 * JNI surface to veilid-core.
 *
 * SKELETON: the native side ([veilid_bridge.cpp]) is a stub that returns a
 * not-implemented sentinel. The real implementation links veilid-core and is
 * built host-side (see ../../../../BUILD_RUNBOOK.md). These signatures ARE the
 * contract the native layer must satisfy; the Kotlin client codes against them.
 *
 * Return conventions:
 *  - `Long` node handle: a positive handle on success, negative on error.
 *  - `Int`: 0 on success, negative on error.
 *  - `ByteArray?` / `String?`: the value, or null when unavailable. For
 *    [appCall], null specifically means "no reply" (the desktop is under Bulbe
 *    or the route is down) and the caller must treat it as terminal, not success.
 */
object VeilidBridge {
    init {
        System.loadLibrary("optioignon_veilid")
    }

    /** Initialise the node from a JSON config. Returns a node handle or <0. */
    external fun nodeInit(configJson: String): Long

    /** Attach the node to the network. Returns 0 on success. */
    external fun nodeAttach(handle: Long): Int

    /** Detach the node from the network. Returns 0 on success. */
    external fun nodeDetach(handle: Long): Int

    /** Shut the node down and release the handle. Returns 0 on success. */
    external fun nodeShutdown(handle: Long): Int

    /** Allocate a private route; returns the blob to hand the desktop, or null. */
    external fun routeAllocate(handle: Long): ByteArray?

    /** Import the desktop's route blob; returns a route id, or null. */
    external fun routeImport(handle: Long, blob: ByteArray): String?

    /**
     * Send an app_call over a route and return the reply bytes (the core RPC).
     *
     * [requestBytes] and the reply are the encoded wire envelopes from the sync
     * contract. Returns null on transport failure / no reply.
     */
    external fun appCall(handle: Long, routeId: String, requestBytes: ByteArray): ByteArray?

    /** Open a DHT record for sync. Returns 0 on success. */
    external fun recordOpen(handle: Long, key: String): Int

    /** Read a DHT record subkey. Returns the bytes, or null. */
    external fun recordGet(handle: Long, key: String, subkey: Int): ByteArray?

    /** Write a DHT record subkey. Returns 0 on success. */
    external fun recordSet(handle: Long, key: String, subkey: Int, data: ByteArray): Int

    /** Close a DHT record. Returns 0 on success. */
    external fun recordClose(handle: Long, key: String): Int
}
