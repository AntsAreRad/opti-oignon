package org.optioignon.mobile.wire

import kotlinx.serialization.SerializationException
import kotlinx.serialization.encodeToString
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for the wire envelopes. These run on a JVM: no device, no
 * emulator, no native library. Everything asserted here is behaviour of the
 * codec or of the declared defaults, never a value this file just wrote and
 * read back through nothing.
 *
 * The desktop responder is the other half of every assertion below. When one
 * of these fails, the phone and the desktop have stopped agreeing about the
 * wire, which is a fault no compiler on either side would have reported.
 */

class WireTest {
    @Test
    fun protocolVersionIsTheOneTheResponderExpects() {
        assertEquals(1, Wire.PROTOCOL_VERSION)
    }

    @Test
    fun unknownKeysAreRejectedRatherThanIgnored() {
        val foreign = """{"v":1,"type":"remote_infer","device":"d",""" +
            """"request_id":"r","prompt":"p","surprise":true}"""
        var refused = false
        try {
            Wire.json.decodeFromString<InferRequest>(foreign)
        } catch (expected: SerializationException) {
            refused = true
        }
        assertTrue(
            "an unknown key must be refused; a surface that ignores what it " +
                "does not understand is not closed",
            refused,
        )
    }
}

class InferRequestTest {
    @Test
    fun aRequestSurvivesTheRoundTripUnchanged() {
        val sent = InferRequest(
            device = "phone-1",
            requestId = "req-1",
            prompt = "hello",
            rag = true,
        )
        val back = Wire.json.decodeFromString<InferRequest>(
            Wire.json.encodeToString(sent),
        )
        assertEquals(sent, back)
    }

    @Test
    fun anAbsentRagCarriesNoFieldAtAll() {
        val encoded = Wire.json.encodeToString(
            InferRequest(device = "d", requestId = "r", prompt = "p"),
        )
        assertFalse(
            "explicitNulls = false must omit the field, not send a null",
            encoded.contains("rag"),
        )
        assertNull(Wire.json.decodeFromString<InferRequest>(encoded).rag)
    }

    @Test
    fun theTypeDiscriminatorIsTheInitialOne() {
        assertEquals(
            Wire.TYPE_INFER,
            InferRequest(device = "d", requestId = "r", prompt = "p").type,
        )
    }
}

class ContRequestTest {
    @Test
    fun aContinuationSurvivesTheRoundTripUnchanged() {
        val sent = ContRequest(device = "phone-1", requestId = "req-1", cursor = 7)
        val back = Wire.json.decodeFromString<ContRequest>(
            Wire.json.encodeToString(sent),
        )
        assertEquals(sent, back)
        assertEquals(Wire.TYPE_INFER_CONT, back.type)
    }
}

class InferReplyTest {
    @Test
    fun aSuccessAndARefusalAreDisjointOnTheWire() {
        val success = Wire.json.decodeFromString<InferReply>(
            """{"v":1,"type":"remote_infer","ok":true,"request_id":"r",""" +
                """"content":"hi","cursor":2,"done":true}""",
        )
        assertTrue(success.ok)
        assertFalse(success.refused)
        assertNull(
            "a success carries no reason; a reply that is both is neither",
            success.reason,
        )

        val refusal = Wire.json.decodeFromString<InferReply>(
            """{"v":1,"type":"remote_infer","ok":false,"refused":true,""" +
                """"request_id":"r","reason":"rate_limited"}""",
        )
        assertFalse(refusal.ok)
        assertTrue(refusal.refused)
        assertEquals(Reason.RATE_LIMITED, refusal.reason)
    }

    @Test
    fun theAbsentFieldsOfEachShapeFallBackRatherThanFail() {
        val minimal = Wire.json.decodeFromString<InferReply>("""{"ok":true}""")
        assertEquals("", minimal.content)
        assertEquals(0, minimal.cursor)
        assertFalse(minimal.done)
    }
}

class ReasonTest {
    @Test
    fun everyReasonIsDistinctAndNonEmpty() {
        val all = listOf(
            Reason.MALFORMED,
            Reason.PROVENANCE_MISMATCH,
            Reason.NO_AUTHENTICATED_IDENTITY,
            Reason.UNKNOWN_DEVICE,
            Reason.PEER_NOT_CONFIRMED,
            Reason.REMOTE_CHAT_DISABLED,
            Reason.RAG_NOT_GRANTED,
            Reason.OUT_OF_SURFACE,
            Reason.RATE_LIMITED,
            Reason.EXECUTION_ERROR,
            Reason.BUFFER_MISMATCH,
        )
        assertEquals(
            "two reasons sharing a string would be indistinguishable to the " +
                "phone, which decides what to retry from this value",
            all.size,
            all.toSet().size,
        )
        assertFalse(all.any { it.isEmpty() })
    }
}
