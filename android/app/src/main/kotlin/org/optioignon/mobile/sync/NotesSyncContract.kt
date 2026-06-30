package org.optioignon.mobile.sync

/**
 * Notes sync, phone side (see the project's MOBILE_SYNC_CONTRACT.md).
 *
 * The phone is a CONSUMER. There is deliberately NO API here to set the per-note
 * phone-sync opt-in: that is a desktop-only human trust decision, it is not on
 * the sync wire (the outbound payload omits the flag), and the phone has no
 * message that flips it. Sync is watermark pull; the desktop filters
 * phone-allowed notes at serve time against a live lookup.
 *
 * This type exists to make that contract explicit in code: it offers a pull by
 * watermark and nothing that could ever opt a note in.
 *
 * SKELETON: the record transport is the JNI bridge stub; the pull contract is
 * validated host-side.
 */
object NotesSyncContract {
    /**
     * The phone advances this watermark across pulls. The desktop serves only
     * notes it has opted in for the phone, beyond this watermark. A note newly
     * opted in at the desktop is republished full-state, so it arrives on the
     * next pull even if the watermark is already past its original position.
     */
    data class PullCursor(val watermark: Long)

    // Intentionally NO setMobileAllowed / optIn surface. See the type doc above.
}
