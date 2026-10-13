"""Contracts of the tone shifter: what its rules read, and what comes back.

The rules of a mode and the custom rules read the prose of a reply, never its code. The code blocks used to stand
in the text, while the rules ran, as markers of the form __CODE_BLOCK_<n>__, put back afterwards wherever they
appeared: a model that wrote the marker outside the code had a block copied there, once per marker, and a block that
quoted another block's marker had that block put inside it. The reply is now read as its code blocks and the prose
between them, and only the prose is given to the rules.

  * TS1 -- a code block comes back as it was, character for character: no rule reads it, nothing the model writes
    stands for one, and no block is put back inside another.
  * TS2 -- a word that touches a code block is a word to the rules.
  * TS3 -- a reply with no code block is transformed exactly as the rules read the whole of it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _isolation import isolate, source  # noqa: E402

BUDGET_S = {
    "test_ts1_a_code_block_comes_back_as_it_was_and_nothing_the_model_writes_stands_for_one": 3.0,
    "test_ts2_a_word_that_touches_a_code_block_is_a_word_to_the_rules": 3.0,
    "test_ts3_a_reply_with_no_code_block_is_transformed_as_the_rules_read_the_whole_of_it": 3.0,
}


def _shifter():
    """(tone shifter, restore): the plugin's module and the check its custom rules use, loaded in one window."""
    loaded, restore = isolate(targets={"opti_oignon.linear_pattern": source("linear_pattern.py"),
                                       "opti_oignon.linear_engine": source("linear_engine.py"),
                                       "tone_shifter_entry_point": source("plugins", "tone-shifter", "entry_point.py")})
    return loaded["tone_shifter_entry_point"], restore


# ---------------------------------------------------------------------------
# TS1 -- a code block comes back as it was
# ---------------------------------------------------------------------------
def test_ts1_a_code_block_comes_back_as_it_was_and_nothing_the_model_writes_stands_for_one():
    quoting, quoted = "`__CODE_BLOCK_1__`", "`real`"
    fence = "```\nIt is x = 1  # It is\n```"
    text = ("Use " + quoting + " then " + quoted + ". It is plain.\n" + "See __CODE_BLOCK_2__ here. " * 3 + "\n" + fence)
    shifter, restore = _shifter()
    try:
        out = shifter.transform_tone(text, "academic", custom_rules={"then": "and then"})
    finally:
        restore()
    assert "It appears to be plain" in out and " and then " in out, ("control: the rules read the prose", out)
    assert quoting in out and quoted in out, ("a block that quotes another's marker comes back as it was", out)
    assert out.count(fence) == 1, ("no block is copied where the model wrote a marker", out.count(fence))
    assert out.count("__CODE_BLOCK_2__") == 3, ("the marker the model wrote stays prose", out)
    assert fence in out, "no rule reads a code block (its 'It is' stays)"


# ---------------------------------------------------------------------------
# TS2 -- a word that touches a code block is a word to the rules
# ---------------------------------------------------------------------------
def test_ts2_a_word_that_touches_a_code_block_is_a_word_to_the_rules():
    shifter, restore = _shifter()
    try:
        out = shifter.transform_tone("It is`x` and so it is.", "academic")
    finally:
        restore()
    assert out.startswith("It appears to be`x`"), out


# ---------------------------------------------------------------------------
# TS3 -- a reply with no code block is transformed as before
# ---------------------------------------------------------------------------
def test_ts3_a_reply_with_no_code_block_is_transformed_as_the_rules_read_the_whole_of_it():
    texts = ["It is very good. However, we do not know.", "Basically, it is etc. e.g. this vs. that",
             "I am going to use it, i.e. the thing.\nNew line here. Don't stop.", ""]
    custom = {"good": "fine", r"\bthe\b": "THE"}
    shifter, restore = _shifter()
    try:
        changed = 0
        for mode, rules in shifter.MODE_RULES.items():
            for text in texts:
                out = shifter.transform_tone(text, mode)
                assert out == shifter.apply_rules(text, rules), (mode, text, out)
                changed += out != text
        for text in texts:
            assert shifter.transform_tone(text, "none", custom_rules=custom) == \
                shifter.apply_custom_rules(text, custom), text
        assert shifter.transform_tone(texts[0], "none", custom_rules=custom) != texts[0], "control: a custom rule reads"
    finally:
        restore()
    assert changed >= len(shifter.MODE_RULES), ("control: every mode changes some text", changed)
