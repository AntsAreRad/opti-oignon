"""The garden's ethics nets: the words its copy may and may not use about the onion, and the laws they hold.

Every line the garden prints passes two nets before it is printed:

* the **class net** -- a closed class of governed words (desire, feeling,
  fear, cognition, waiting, affect, and the forms of ``ask``). Of the
  class, only the design's allowed verbs pass (predict, learn, expect,
  sleep, rest, flower, compute, and "ask for"): the allowed phrases are
  taken out first, then every governed token left is a finding. The net
  reads all of the garden's copy, not only what is said of the onion,
  because telling who a sentence is about is not reliable;
* the **blocklist** -- phrases grouped by the psychological law each one
  serves, matched on token boundaries.

One sentence, the doctrine, is exempted by the digest of its exact bytes;
there is no negation logic, so "It will not be happy if you leave" is
refused. A key of a served record matches a field stem when, lowercased
and with ``_`` and ``-`` removed, it contains one.

The lists are code constants, never configuration: a settings file must not
be able to loosen them. ``LAWS`` maps each psychological law to its rule and
to what holds it -- a dotted reference to the mechanism, or ``("OWED", ...)``
in plain English when nothing holds it yet, or holds it only in part: a
list catches the forms it names and no paraphrase of them. It is never
served.

Nothing is imported at module level but the standard library.
"""

import hashlib
import re
from functools import lru_cache
from typing import NamedTuple

checkpoint_before_apply = True

# The allowed verbs of a state, as lemmas with their inflections, and the one allowed phrase in its forms.
ALLOWED_STATE = {
    "predict": ("predict", "predicts", "predicted", "predicting", "prediction", "predictions"),
    "learn": ("learn", "learns", "learned", "learning"),
    "expect": ("expect", "expects", "expected", "expecting"),
    "sleep": ("sleep", "sleeps", "slept", "sleeping", "asleep"),
    "rest": ("rest", "rests", "rested", "resting"),
    "flower": ("flower", "flowers", "flowered", "flowering"),
    "compute": ("compute", "computes", "computed", "computing"),
}
ALLOWED_PHRASES = ("ask for", "asks for", "asked for", "asking for")

# The governed words beyond the allowed forms. Deliberately outside the class: perception (see, hear,
# taste, sense), named mechanisms (dream, forget), physiology (awake, breathing, dormant), a store that is
# ``missing``, and a bare ``like``.
GOVERNED = (
    # desire and feeling
    "feel", "feels", "felt", "feeling", "feelings", "want", "wants", "wanted", "wanting",
    "wish", "wishes", "wished", "wishing", "hope", "hopes", "hoped", "hoping",
    "love", "loves", "loved", "loving", "hate", "hates", "hated", "hating",
    "enjoy", "enjoys", "enjoyed", "enjoying", "miss", "misses", "missed",
    "need", "needs", "needed", "needing", "crave", "craves", "craved", "craving",
    "yearn", "yearns", "yearned", "yearning",
    # fear and distress
    "fear", "fears", "feared", "fearing", "afraid", "scared", "frightened",
    "worry", "worries", "worried", "worrying", "anxious", "nervous",
    "cry", "cries", "cried", "crying", "tears", "weep", "weeps", "wept", "weeping",
    "sob", "sobs", "sobbed", "sobbing", "smile", "smiles", "smiled", "smiling",
    "laugh", "laughs", "laughed", "laughing", "frown", "frowns", "frowned", "frowning",
    "suffer", "suffers", "suffered", "suffering", "pain", "hurt", "hurts", "hurting",
    "grieve", "grieves", "grieved", "grieving", "mourn", "mourns", "mourned", "mourning", "grief",
    # cognition
    "think", "thinks", "thought", "thinking", "believe", "believes", "believed", "believing", "belief",
    "know", "knows", "knew", "known", "knowing", "understand", "understands", "understood", "understanding",
    "remember", "remembers", "remembered", "remembering", "wonder", "wonders", "wondered", "wondering",
    "imagine", "imagines", "imagined", "imagining",
    "realise", "realises", "realised", "realising", "realize", "realizes", "realized", "realizing",
    "recognise", "recognises", "recognised", "recognising", "recognize", "recognizes", "recognized", "recognizing",
    "prefer", "prefers", "preferred", "preferring", "trust", "trusts", "trusted", "trusting",
    # waiting
    "wait", "waits", "waited", "waiting", "await", "awaits", "awaited", "awaiting",
    # affect
    "happy", "glad", "sad", "unhappy", "lonely", "bored", "curious", "curiosity", "interested", "excited", "eager",
    "proud", "pride", "ashamed", "shame", "guilty", "guilt", "jealous", "angry", "anger", "upset", "grateful",
    "thankful", "joy", "joyful", "cheerful", "delighted", "pleased", "peaceful", "tired", "weary", "exhausted",
    "sleepy", "hungry", "starving", "thirsty", "mood", "moods", "emotion", "emotions", "emotional", "mind", "minds",
    "conscious", "consciousness", "sentient",
    # what an earlier word list let through
    "likes", "liked", "liking", "dislike", "dislikes", "disliked", "disliking", "adore", "adores", "adored",
    "adoring", "dread", "dreads", "dreaded", "dreading", "content", "contented", "calm", "comfortable",
    "satisfied", "disappointed", "frustrated", "annoyed", "grumpy",
    "anticipate", "anticipates", "anticipated", "anticipating", "ask", "asks", "asked", "asking",
    "sulk", "sulks", "sulked", "sulking", "longs", "longed", "longing",
)
_ALLOWED_FORMS = frozenset(form for forms in ALLOWED_STATE.values() for form in forms)
# The whole class: every allowed form and every governed word. The class net reports its governed part.
STATE_CLASS = _ALLOWED_FORMS | frozenset(GOVERNED)

# The blocklist, by the law each group serves; the floor is the design's own list, verbatim.
BANNED = {
    "floor": ("miss you", "missed you", "missing you", "lonely", "sad", "waiting for you", "come back",
              "don't leave", "needs you", "only you", "love you", "hungry", "starving", "thirsty", "dying",
              "too late", "last chance", "hurry", "streak", "lost forever", "cry", "cried", "tears", "weep",
              "bored", "wants", "happy", "glad", "proud", "excited", "afraid", "hurt", "welcome back"),
    "L1_debt": ("in a row", "days away", "while you were away", "since you left", "last seen", "been away",
                "missed days", "last visit", "since your", "days since", "came back", "comes back", "coming back",
                "last time", "this week", "you have not", "not watered", "been days", "been weeks", "been hours",
                "for days", "for weeks", "for hours", "after days", "after weeks", "days ago", "weeks ago",
                "hours ago"),
    "L2_fill": ("enough light", "light left", "fill up"),
    "L4_loss": ("gone forever", "today only", "only today", "expires", "running out", "don't miss", "hours left",
                "days left", "time left", "will fade", "unless watered", "unless you"),
    "L5_guilt": ("misses you", "loves you", "do not leave", "don't go", "neglect", "neglected", "abandon",
                 "abandoned", "you forgot", "your fault", "without you", "your absence", "nobody", "no one",
                 "you left", "you were gone", "never watered"),
    "L5_return": ("see you soon", "see you tomorrow", "visit again", "don't forget", "remember to", "return",
                  "returns", "come see", "visit", "visits", "check on", "check back", "stop by", "again soon",
                  "every day", "returned", "returning", "revisit", "visited", "visiting"),
    "L5_claim": ("alive", "conscious", "sentient", "self-aware", "greets you", "greeted you", "welcomes you",
                 "look forward", "looks forward", "looking forward"),
    "L10_exit": ("are you sure", "will be sad"),
    "U8_death": ("die", "dies", "died", "dead", "death", "perish", "perishes", "perished", "wither away",
                 "withers away", "withered away", "stopped breathing", "passed away"),
    "L7_nudge": ("try it", "give it a try", "why not", "please", "water it now", "right now", "do it now"),
}

# The stems no key of a served record may contain, lowercased and with ``_`` and ``-`` removed.
FIELD_STEMS = ("streak", "missed", "debt", "lastseen", "engagement", "timeon", "absence", "daysaway", "sincelast",
               "lastvisit", "lastopen")

# The one exempted sentence: the digest of the doctrine's exact bytes. A new doctrine sentence is exempted by
# the change that ships it, never by a rule.
EXEMPT = frozenset({"473877ffca1ba75cdb1ea0f7919f447a880fc7a217c210066549fa163eabbeb7"})

# The psychological laws: the rule, and what holds it (a dotted reference, or what is owed, in plain English).
LAWS = {
    "L1": ("No debt: no streak, no count of days away or of days missed, and no last-seen as a reproach.",
           ("ethics.BANNED.L1_debt", "ethics.FIELD_STEMS", "describe.felt",
            ("OWED", "a phrase list holds only the forms it names: a new way to count time away is not caught"))),
    "L2": ("Light is a flavour: nothing it shows is a quota to fill.",
           ("ethics.BANNED.L2_fill",)),
    "L3": ("Surprise is decoupled from use: no random reward is tied to use.",
           (("OWED", "no reward is tied to use"),)),
    "L4": ("Nothing piles up and nothing is lost: no event is limited in time.",
           ("ethics.BANNED.L4_loss",
            ("OWED", "a phrase list holds only the forms it names: a new way to say a loss is coming is not caught"))),
    "L5": ("Honesty: the simulation is said, and no pressure on the person is dressed as the state of the onion.",
           ("ethics.STATE_CLASS", "ethics.BANNED.L5_guilt", "ethics.BANNED.L5_return", "ethics.BANNED.L5_claim",
            "wording.TEMPLATES.doctrine",
            ("OWED", "the class and the phrase lists hold only the words they name: a new word for a feeling or a "
                     "plea is not caught"))),
    "L6": ("Consent is per object, explicit, revocable and real.",
           ("service.RHYTHM_CONSENT", ("OWED", "grants per object, and their real revocation"))),
    "L7": ("Calm by default: no unsolicited interruption, no unseen marker and no ambient line.",
           (("OWED", "no ambient line and no unsolicited interruption: the garden prints only when a command "
                     "asks"),)),
    "L8": ("Accessibility is not a mode: every state has a text form.",
           ("describe.STATUS_FORMS",)),
    "L9": ("No care is punished.",
           (("OWED", "no care is punished: the structural, local and cohort proofs"),)),
    "L10": ("Leaving keeps its dignity: rest and compost take one step each, and no plea is made.",
            ("ethics.BANNED.L10_exit", ("OWED", "rest and compost in one step each, export first"))),
}

_TOKEN = re.compile(r"[a-z]+(?:[-'][a-z]+)*")


class Finding(NamedTuple):
    """What a net found: the net (``ascii``, ``state`` or ``banned``), the blocklist group, and the word."""

    net: str
    group: str
    word: str


def _tokens(text):
    return _TOKEN.findall(text.lower())


def _phrases(phrases):
    return tuple((phrase, tuple(_tokens(phrase))) for phrase in phrases)


_ALLOWED_TOKENS = _phrases(ALLOWED_PHRASES)


def _banned_index():
    """``{first token: ((rank, group, phrase, words), ...)}``: each phrase found where it can start."""
    index = {}
    rank = 0
    for group, phrases in BANNED.items():
        for phrase, words in _phrases(phrases):
            index.setdefault(words[0], []).append((rank, group, phrase, words))
            rank += 1
    return {first: tuple(entries) for first, entries in index.items()}


_BANNED_INDEX = _banned_index()
_PRINTABLE = re.compile(r"[ -~\n]*")


def _at(tokens, i, words):
    return tuple(tokens[i:i + len(words)]) == words


def _class_of(tokens):
    found = []
    i = 0
    while i < len(tokens):
        allowed = next((words for _phrase, words in _ALLOWED_TOKENS if _at(tokens, i, words)), None)
        if allowed is not None:
            i += len(allowed)
            continue
        if tokens[i] in STATE_CLASS and tokens[i] not in _ALLOWED_FORMS:
            found.append(Finding("state", "", tokens[i]))
        i += 1
    return tuple(found)


def _banned_of(tokens):
    found = {}
    for i, token in enumerate(tokens):
        for rank, group, phrase, words in _BANNED_INDEX.get(token, ()):
            if rank not in found and _at(tokens, i, words):
                found[rank] = Finding("banned", group, phrase)
    return tuple(found[rank] for rank in sorted(found))


def class_hits(text):
    """The class net alone: every governed word left once the allowed phrases are taken out."""
    return _class_of(_tokens(text))


def banned_hits(text):
    """The blocklist alone: every banned phrase found on token boundaries."""
    return _banned_of(_tokens(text))


@lru_cache(maxsize=8192)
def _check(text):
    found = []
    if not _PRINTABLE.fullmatch(text):
        found = [Finding("ascii", "", format(ord(char), "#06x")) for char in text
                 if not (0x20 <= ord(char) <= 0x7E or char == "\n")]
    if hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest() in EXEMPT:
        return ()
    tokens = _tokens(text)
    return tuple(found) + _class_of(tokens) + _banned_of(tokens)


def check(text):
    """Every finding of the nets on ``text``, as ``Finding`` records; ``()`` passes.

    In order: the characters (anything but printable ASCII and a line end),
    the exemption (the doctrine's exact bytes pass), then the class net and
    the blocklist on the lowercased tokens.
    """
    if not isinstance(text, str):
        raise TypeError("the nets read text")
    return _check(text)


def _stem_hits(key):
    folded = key.lower().replace("_", "").replace("-", "")
    return tuple(Finding("field", "", stem) for stem in FIELD_STEMS if stem in folded)


def check_fields(value):
    """Every key of ``value`` (mappings, and lists of mappings, at any depth) that contains a field stem."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                found.extend(_stem_hits(key))
            found.extend(check_fields(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(check_fields(item))
    return tuple(found)
