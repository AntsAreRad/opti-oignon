package org.optioignon.mobile.wire

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Wire envelopes for remote inference. Every field name here is the desktop
 * responder's literal contract (see ../../../../../../MOBILE_SYNC_CONTRACT.md);
 * the phone MUST speak exactly these.
 */
object Wire {
    const val PROTOCOL_VERSION = 1
    const val TYPE_INFER = "remote_infer"
    const val TYPE_INFER_CONT = "remote_infer_cont"

    /**
     * Strict codec. `explicitNulls = false` omits a null `rag` from the wire (so
     * a request without RAG carries no `rag` field at all). Unknown keys are
     * rejected: the surface on both sides is closed.
     */
    val json = Json {
        ignoreUnknownKeys = false
        encodeDefaults = true
        explicitNulls = false
    }
}

/** Initial request: the only fields the tier 1 surface accepts, plus optional RAG. */
@Serializable
data class InferRequest(
    @SerialName("v") val v: Int = Wire.PROTOCOL_VERSION,
    @SerialName("type") val type: String = Wire.TYPE_INFER,
    @SerialName("device") val device: String,
    @SerialName("request_id") val requestId: String,
    @SerialName("prompt") val prompt: String,
    @SerialName("rag") val rag: Boolean? = null,
)

/** Continuation request: request id and cursor, bound to this device. */
@Serializable
data class ContRequest(
    @SerialName("v") val v: Int = Wire.PROTOCOL_VERSION,
    @SerialName("type") val type: String = Wire.TYPE_INFER_CONT,
    @SerialName("device") val device: String,
    @SerialName("request_id") val requestId: String,
    @SerialName("cursor") val cursor: Int,
)

/**
 * A reply, success or refusal. The two shapes are disjoint on the wire; the
 * defaults below absorb whichever fields the matching shape omits. `ok` and
 * `refused` discriminate: a success carries `ok = true`; a refusal carries
 * `ok = false, refused = true` plus `reason` / `detail`.
 */
@Serializable
data class InferReply(
    @SerialName("v") val v: Int = Wire.PROTOCOL_VERSION,
    @SerialName("type") val type: String = "",
    @SerialName("ok") val ok: Boolean = false,
    @SerialName("refused") val refused: Boolean = false,
    @SerialName("request_id") val requestId: String = "",
    // success fields
    @SerialName("content") val content: String = "",
    @SerialName("cursor") val cursor: Int = 0,
    @SerialName("done") val done: Boolean = false,
    // refusal fields
    @SerialName("reason") val reason: String? = null,
    @SerialName("detail") val detail: String? = null,
)

/** The complete refusal reason set the desktop emits. */
object Reason {
    const val MALFORMED = "malformed"
    const val PROVENANCE_MISMATCH = "provenance_mismatch"
    const val NO_AUTHENTICATED_IDENTITY = "no_authenticated_identity"
    const val UNKNOWN_DEVICE = "unknown_device"
    const val PEER_NOT_CONFIRMED = "peer_not_confirmed"
    const val REMOTE_CHAT_DISABLED = "remote_chat_disabled"
    const val RAG_NOT_GRANTED = "rag_not_granted"
    const val OUT_OF_SURFACE = "out_of_surface"
    const val RATE_LIMITED = "rate_limited"
    const val EXECUTION_ERROR = "execution_error"
    const val BUFFER_MISMATCH = "buffer_mismatch"
}
