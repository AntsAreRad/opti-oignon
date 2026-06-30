package org.optioignon.mobile.sync

import org.optioignon.mobile.wire.ContRequest
import org.optioignon.mobile.wire.InferReply
import org.optioignon.mobile.wire.InferRequest
import org.optioignon.mobile.wire.Wire
import org.optioignon.veilid.VeilidBridge

/**
 * Client side of the remote-inference contract (see the project's
 * MOBILE_SYNC_CONTRACT.md). One chat turn is: send the initial request, then
 * pull successive chunks until `done`.
 *
 * SKELETON: the transport calls into [VeilidBridge], whose native side is a stub
 * in this tree. The loop, the envelope handling, and the refusal mapping ARE the
 * contract and are validated host-side against a real desktop responder.
 */
class RemoteInferenceClient(
    private val nodeHandle: Long,
    private val routeId: String,
    private val deviceId: String,
) {
    /** The result of one chat turn. */
    sealed interface Outcome {
        data class Completed(val text: String) : Outcome
        data class Refused(val reason: String, val detail: String?) : Outcome

        /**
         * No reply. The desktop is under Bulbe (it sends nothing) or the route is
         * down. Terminal and non-retryable on this route; never treated as success.
         */
        data object NoReply : Outcome
    }

    /**
     * Run one remote chat turn. [requestId] must be fresh and unique. Set
     * [requestRag] only if the desktop has granted this device the RAG sub-grant;
     * otherwise the desktop refuses with `rag_not_granted`.
     */
    fun chat(prompt: String, requestId: String, requestRag: Boolean = false): Outcome {
        val firstJson = Wire.json.encodeToString(
            InferRequest.serializer(),
            InferRequest(
                device = deviceId,
                requestId = requestId,
                prompt = prompt,
                rag = if (requestRag) true else null,
            ),
        )
        var reply = call(firstJson) ?: return Outcome.NoReply

        val assembled = StringBuilder()
        while (true) {
            if (reply.refused) {
                return Outcome.Refused(reply.reason ?: "unknown", reply.detail)
            }
            assembled.append(reply.content)
            if (reply.done) {
                return Outcome.Completed(assembled.toString())
            }
            // Pull the next chunk with the cursor the desktop returned. The
            // continuation does not consume the device's rate budget.
            val contJson = Wire.json.encodeToString(
                ContRequest.serializer(),
                ContRequest(device = deviceId, requestId = requestId, cursor = reply.cursor),
            )
            reply = call(contJson) ?: return Outcome.NoReply
        }
    }

    /** Send one envelope; decode the reply, or null on no reply. */
    private fun call(requestJson: String): InferReply? {
        val replyBytes = VeilidBridge.appCall(
            nodeHandle, routeId, requestJson.encodeToByteArray(),
        ) ?: return null
        return Wire.json.decodeFromString(InferReply.serializer(), replyBytes.decodeToString())
    }
}
