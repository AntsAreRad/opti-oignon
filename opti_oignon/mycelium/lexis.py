"""Words for the full-text index and its questions: an identifier kept whole AND cut into its words, so
`getUserName`, `get_user_name` and "user name" meet; stop words of French and English left out of questions,
compared with their accents folded. FTS5 folds case and accents and stems English words itself (porter over
unicode61, `_` kept inside a token)."""
import re
import unicodedata

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

WORD = re.compile(r"\w+", re.UNICODE)
SUBWORD = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")
JOINED = re.compile(r"_|[a-z0-9][A-Z]|[A-Z][A-Z][a-z]")
STOP = set("""a an and are as at be by for from has have how i if in into is it its of on or that the this to was what
when where which who why will with does do can should would could my your their there then than so not no yes
le la les un une des du de d l et ou est sont en au aux ce cet cette ces se sa son ses qui que quoi quel quelle quels
quelles comment pourquoi quand il elle ils elles on je tu nous vous y pour par sur dans avec sans pas ne plus
moins ca cela fait faire peut peux etre avoir mon ma mes ton ta tes notre nos leur leurs si alors donc
mais""".split())


def pieces(token):
    """The words inside an identifier (`get_user_name`, `getUserName`, `HTTPServer`); nothing for a plain word,
    accented or not."""
    if not JOINED.search(token):
        return []
    found = SUBWORD.findall(token.replace("_", " "))
    return found if len(found) > 1 else []


def subwords(text):
    """The words inside the identifiers of ``text``, space-separated: the ``words`` column of the index."""
    out = []
    for token in WORD.findall(text):
        out.extend(pieces(token))
    return " ".join(out)


def identifiers(text):
    """The tokens of a question that look like code: joined words (`snake_case`, `camelCase`) or dotted names."""
    found = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", text):
        last = token.split(".")[-1]
        if (JOINED.search(last) or "." in token) and last not in found:
            found.append(last)
    return found


def question_terms(query, stop=STOP):
    """The question's terms for a MATCH: its words (folded by FTS5 itself), stop words off, and the words inside
    its identifiers as well."""
    terms = []
    for token in WORD.findall(query):
        for word in [token] + pieces(token):
            folded = word.lower()
            bare = "".join(c for c in unicodedata.normalize("NFKD", folded) if not unicodedata.combining(c))
            if len(folded) < 2 or bare in stop or (folded.isdigit() and len(folded) < 3):
                continue
            if folded not in terms:
                terms.append(folded)
    return terms


def match_expression(terms):
    """Terms OR-ed, each quoted: a part needs one of them, bm25 ranks by how many rare ones it holds."""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
