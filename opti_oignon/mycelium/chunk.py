"""Cutting a file into parts by its syntax, with what each part defines and what it names.

A part is the unit the Mycelium finds and cites: (kind, symbol, first line, last line), its definitions (symbols
another part may name) and its references (names it uses: calls, imports, setting keys in string literals, routes,
identifiers between backticks). The map (``graph``) is the join of references and definitions, so nothing is
resolved at cutting time and a changed file never leaves a stale link behind.

  Python      by ``ast``: the module's head (docstring, imports, constants) and every run of module-level code
              between definitions, each function, each class head, each method; nothing of a file is left out.
  Markdown    by headings (a section's symbol is its heading path).
  YAML        by top-level key, WITH the comment block above it (oo's config files are commented in prose).
  TS, JS, Svelte, Rust   by top-level declarations, braces counted.
  anything else          windows of WINDOW lines overlapping by OVERLAP.
A unit longer than MAX_LINES is cut into windows; its first window carries its definitions and references."""
import ast
import re
from dataclasses import dataclass, field

# New-module safety rule: any change this module drives through the system
# must checkpoint first. Hardcoded, never overridable.
checkpoint_before_apply = True

WINDOW, OVERLAP, MAX_LINES = 40, 10, 60
# A part shorter than this is merged with its short neighbours (up to WINDOW lines): a one-line key or a three-line
# getter is too little to find by its words, and its vector matches any question.
MIN_LINES = 6
# Raised when what a part is or holds changes: a store cut by another version cuts its files again (its vectors stay
# for every text that did not change).
CHUNKER_VERSION = 5
LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log"}
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
BACKTICK = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)(?:\(\))?`")
KEYLIKE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
ROUTE = re.compile(r"^/api/[A-Za-z0-9_\-/{}.$:]*$")


@dataclass
class Part:
    kind: str
    symbol: str
    first: int
    last: int
    defines: list = field(default_factory=list)     # (name, qualname, kind)
    refs: list = field(default_factory=list)        # (name, kind, module)
    said: list = field(default_factory=list)        # what the part says to a person: raised, logged, printed


def language(path):
    p = path.lower()
    for suffix, lang in ((".py", "python"), (".md", "markdown"), (".yaml", "yaml"), (".yml", "yaml"),
                         (".ts", "ts"), (".js", "ts"), (".mjs", "ts"), (".cjs", "ts"), (".svelte", "svelte"),
                         (".rs", "rust")):
        if p.endswith(suffix):
            return lang
    return "text"


def windows(first, last):
    out, start = [], first
    while start <= last:
        end = min(last, start + WINDOW - 1)
        out.append((start, end))
        if end == last:
            break
        start = end - OVERLAP + 1
    return out


def route_of(text):
    """A route path, its parameters written ``{}`` whatever their spelling (`{id}`, `${id}`, `:id`)."""
    # [^{}]: a scan stops at the next brace, so no two attempts read the same text (linear on any input).
    text = re.sub(r"\$\{[^{}]*\}|\{[^{}]*\}|:[A-Za-z_]\w*", "{}", text.split("?")[0])
    return text.rstrip("/") or "/"


def module_name(path):
    """`opti_oignon/memory/peels.py` -> `opti_oignon.memory.peels`; a package's `__init__.py` -> the package."""
    name = path[:-3].replace("/", ".")
    return name[:-len(".__init__")] if name.endswith(".__init__") else name


def _resolve(module, level, path):
    """The absolute module of a ``from`` import written in ``path``."""
    if not level:
        return module or ""
    package = module_name(path).split(".")
    if not path.endswith("__init__.py"):
        package = package[:-1]
    base = package[:len(package) - (level - 1)] if level > 1 else package
    return ".".join(base + ([module] if module else []))


class _Refs(ast.NodeVisitor):
    """The references of a subtree: calls by name, imported names, key-like strings, routes, backticked names."""

    def __init__(self, path, imported):
        self.path, self.imported, self.out = path, imported, []

    def add(self, name, kind, module=""):
        item = (name, kind, module)
        if item not in self.out:
            self.out.append(item)

    def visit_Call(self, node):
        target = node.func
        if isinstance(target, ast.Name):
            self.add(target.id, "call", self.imported.get(target.id, ""))
        elif isinstance(target, ast.Attribute):
            self.add(target.attr, "call", "")
        self.generic_visit(node)

    def visit_Name(self, node):
        # An imported name used without a call (passed, compared, asserted): a use of what it names.
        if isinstance(node.ctx, ast.Load) and node.id in self.imported:
            self.add(node.id, "use", self.imported[node.id])

    def visit_Import(self, node):
        for alias in node.names:
            self.add(alias.name.split(".")[-1], "import", alias.name)

    def visit_ImportFrom(self, node):
        module = _resolve(node.module, node.level, self.path)
        for alias in node.names:
            self.add(alias.name, "import", module)
        if module:
            self.add(module.split(".")[-1], "import", module)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            text = node.value
            if KEYLIKE.match(text):
                self.add(text, "string")
            elif ROUTE.match(text):
                self.add(route_of(text), "route")
            elif "\n" in text or " " in text:
                for name in BACKTICK.findall(text):
                    self.add(name.split(".")[-1], "mention")

    def visit_JoinedStr(self, node):
        literal = "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
                          for v in node.values)
        if ROUTE.match(literal):
            self.add(route_of(literal), "route")
        self.generic_visit(node)


def _imports(tree, path):
    """{local name: absolute module} of a module's imports, for resolving calls."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = _resolve(node.module, node.level, path)
            for alias in node.names:
                found[alias.asname or alias.name] = module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found[(alias.asname or alias.name).split(".")[0]] = alias.name
    return found


def _refs(nodes, path, imported):
    visitor = _Refs(path, imported)
    for node in nodes:
        visitor.visit(node)
    return visitor.out


def _literal(node):
    """The text of a string literal or f-string (its holes written {}), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
                       for v in node.values)
    return None


def _said(nodes):
    """What a subtree says to a person: the strings of what it raises, logs, prints, warns, or answers as an HTTP
    error's detail. A user reports a bug by these words, not by the code's."""
    out = []

    def keep(node):
        text = _literal(node)
        if text and len(text.strip()) >= 6 and text not in out:
            out.append(text.strip())

    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                for arg in node.exc.args[:2]:
                    keep(arg)
                for kw in node.exc.keywords:
                    if kw.arg in ("detail", "message", "msg", "reason"):
                        keep(kw.value)
            elif isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
                if name in LOG_METHODS or name in ("print", "warn", "HTTPException", "JSONResponse"):
                    for arg in node.args[:2]:
                        keep(arg)
                    for kw in node.keywords:
                        if kw.arg in ("detail", "message", "msg", "content"):
                            keep(kw.value)
    return out


def _route_defs(node):
    """The routes a decorated function serves: `@router.get("/api/x/{id}")` -> `/api/x/{}`."""
    found = []
    for deco in getattr(node, "decorator_list", []):
        if isinstance(deco, ast.Call) and deco.args and isinstance(deco.args[0], ast.Constant) \
                and isinstance(deco.args[0].value, str) and deco.args[0].value.startswith("/"):
            found.append((route_of(deco.args[0].value), route_of(deco.args[0].value), "route"))
    return found


def python_parts(path, lines):
    try:
        tree = ast.parse("\n".join(lines))
    except (SyntaxError, ValueError):
        return None
    imported = _imports(tree, path)
    module = module_name(path)
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def start(node):
        """A definition's first line: its decorators, and the comment lines right above them."""
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        while first > 1 and lines[first - 2].strip().startswith("#"):
            first -= 1
        return first

    out, run = [], []

    def flush(next_start):
        if not run:
            return
        first = 1 if not out else start(run[0])
        last = next_start - 1
        while last > first and not lines[last - 1].strip():
            last -= 1
        head = not out
        doc = run[0] if head and isinstance(run[0], ast.Expr) and isinstance(getattr(run[0], "value", None),
                                                                              ast.Constant) \
            and isinstance(run[0].value.value, str) else None
        if doc is not None:
            # A module's docstring is a part of its own: it is often the design of the module, and mixed with the
            # imports and constants it says less to a search for how something works.
            out.append(Part("doc", module, 1, doc.end_lineno, defines=[(module.split(".")[-1], module, "module")],
                            refs=_refs([doc], path, imported)))
            rest = run[1:]
            if rest:
                first_rest = start(rest[0])
                while last > first_rest and not lines[last - 1].strip():
                    last -= 1
                out.append(Part("module", f"{module}:imports", first_rest, last, refs=_refs(rest, path, imported),
                                said=_said(rest)))
            run.clear()
            return
        out.append(Part("module" if head else "code", module if head else f"{module}:{first}", first, last,
                        defines=[(module.split(".")[-1], module, "module")] if head else [],
                        refs=_refs(run, path, imported), said=_said(run)))
        run.clear()

    for node in tree.body:
        if isinstance(node, defs):
            flush(start(node))
            if not out and start(node) > 1:     # only comments or blank lines above the first definition
                out.append(Part("module", module, 1, start(node) - 1))
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                head_end = (start(methods[0]) - 1) if methods else node.end_lineno
                before = [n for n in node.body if n.lineno <= head_end and n not in methods]
                heading = node.bases + node.decorator_list + before
                out.append(Part("class", node.name, start(node), max(start(node), head_end),
                                defines=[(node.name, node.name, "class")],
                                refs=_refs(heading, path, imported), said=_said(heading)))
                # The class body's statements between and after its methods (constants, aliases): their own parts.
                group = []

                def close(until):
                    if group:
                        last = until
                        while last > group[0].lineno and not lines[last - 1].strip():
                            last -= 1
                        out.append(Part("code", f"{node.name}:{group[0].lineno}", group[0].lineno, last,
                                        refs=_refs(group, path, imported), said=_said(group)))
                        group.clear()

                for item in node.body:
                    if item in methods:
                        close(start(item) - 1)
                        out.append(Part("method", f"{node.name}.{item.name}", start(item), item.end_lineno,
                                        defines=[(item.name, f"{node.name}.{item.name}", "method")]
                                        + _route_defs(item), refs=_refs([item], path, imported), said=_said([item])))
                    elif item.lineno > head_end:
                        group.append(item)
                close(node.end_lineno)
            else:
                out.append(Part("function", node.name, start(node), node.end_lineno,
                                defines=[(node.name, node.name, "function")] + _route_defs(node),
                                refs=_refs([node], path, imported), said=_said([node])))
        else:
            run.append(node)
    flush(len(lines) + 1)
    if not out:
        out.append(Part("module", module, 1, max(1, len(lines))))
    # The module's own name is defined once, by its docstring or head, or by its first part when it opens on a
    # definition.
    holder = next((p for p in out if p.kind in ("doc", "module")), out[0])
    if not any(kind == "module" for _n, _q, kind in holder.defines):
        holder.defines.insert(0, (module.split(".")[-1], module, "module"))
    return out


def markdown_parts(path, lines):
    heads = [i + 1 for i, line in enumerate(lines) if re.match(r"#{1,6} ", line)]
    bounds = ([1] if not heads or heads[0] > 1 else []) + heads
    out, trail = [], []
    for n, first in enumerate(bounds):
        last = (bounds[n + 1] - 1) if n + 1 < len(bounds) else len(lines)
        line = lines[first - 1]
        match = re.match(r"(#{1,6}) (.*)", line)
        if match:
            depth = len(match.group(1))
            trail = trail[:depth - 1] + [match.group(2).strip()]
            symbol = " > ".join(trail)
        else:
            symbol = "(head)"
        text = "\n".join(lines[first - 1:last])
        refs = []
        for name in BACKTICK.findall(text):
            item = (name.split(".")[-1], "mention", "")
            if item not in refs:
                refs.append(item)
        for route in re.findall(r"/api/[A-Za-z0-9_\-/{}.$:]+", text):
            item = (route_of(route), "route", "")
            if item not in refs:
                refs.append(item)
        out.append(Part("section", symbol, first, last, refs=refs))
    return out


def yaml_parts(path, lines):
    """By top-level key, the comment block just above a key going with it."""
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    keys = [i + 1 for i, line in enumerate(lines) if re.match(r"[A-Za-z_][\w\-]*\s*:", line)]
    if not keys:
        return [Part("text", stem, 1, max(1, len(lines)))]
    starts = []
    for k in keys:
        s = k
        while s > 1 and lines[s - 2].startswith("#"):
            s -= 1
        starts.append(s)
    out = []
    if starts[0] > 1:
        out.append(Part("head", stem, 1, starts[0] - 1))
    for n, (s, k) in enumerate(zip(starts, keys)):
        last = (starts[n + 1] - 1) if n + 1 < len(starts) else len(lines)
        while last > k and not lines[last - 1].strip():
            last -= 1
        top = re.match(r"([A-Za-z_][\w\-]*)", lines[k - 1]).group(1)
        defines = [(top, f"{stem}.{top}", "setting")]
        for line in lines[k:last]:
            nested = re.match(r"\s+([A-Za-z_][\w\-]*)\s*:", line)
            if nested and (nested.group(1), f"{stem}.{top}.{nested.group(1)}", "setting") not in defines:
                defines.append((nested.group(1), f"{stem}.{top}.{nested.group(1)}", "setting"))
        out.append(Part("setting", f"{stem}.{top}", s, last, defines=defines))
    return out


DECL = {
    "ts": re.compile(r"^(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?"
                     r"(function\*?|class|interface|type|enum|const|let|var)\s+([A-Za-z_$][\w$]*)"),
    "rust": re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?"
                       r"(fn|struct|enum|trait|impl|mod|type|const|static|macro_rules!)\s*([A-Za-z_][\w]*)?"),
}
# A name starts where no name character precedes it, and a brace list stops at the next brace: each run of text is
# read by one attempt, never by one per position inside it (pattern_cost_guard: linear on any input).
CALL = re.compile(r"(?<![\w$])([A-Za-z_$][\w$]*)\s*\(")
SAYS = re.compile(r"(?:throw\s+new\s+\w*Error|console\.(?:error|warn|log)|(?<!\w)toast\w*|alert|panic!|eprintln!|"
                  r"\.expect)\s*\(\s*['\"`]([^'\"`\n]{6,200})['\"`]")
TS_IMPORT = re.compile(r"import\s+(?:type\s+)?\{([^{}]*)\}\s+from\s+['\"]([^'\"]+)['\"]")
STRING = re.compile(r"['\"`]([^'\"`\n]{3,200})['\"`]")
KEYWORDS = {"if", "for", "while", "switch", "catch", "function", "return", "typeof", "await", "new", "match", "fn",
            "Some", "Ok", "Err", "vec", "println", "format", "assert", "assert_eq", "super", "self", "async"}


def _code_said(text):
    return list(dict.fromkeys(m.strip() for m in SAYS.findall(text)))


def _code_refs(text):
    refs = []

    def add(item):
        if item not in refs:
            refs.append(item)

    for names, source in TS_IMPORT.findall(text):
        for name in names.split(","):
            name = name.strip().split(" as ")[0].strip()
            if name:
                add((name, "import", source))
    for name in CALL.findall(text):
        if name not in KEYWORDS:
            add((name, "call", ""))
    for literal in STRING.findall(text):
        if ROUTE.match(literal):
            add((route_of(literal), "route", ""))
        elif KEYLIKE.match(literal):
            add((literal, "string", ""))
    return refs


def brace_parts(path, lines, lang):
    """Top-level declarations, each until the next one; for Svelte, the script block's declarations (indented by
    at most one level) and the markup in windows."""
    pattern = DECL["rust" if lang == "rust" else "ts"]
    region = range(1, len(lines) + 1)
    markup = []
    if lang == "svelte":
        opens = [i + 1 for i, line in enumerate(lines) if re.match(r"\s*<script\b", line)]
        closes = [i + 1 for i, line in enumerate(lines) if re.match(r"\s*</script>", line)]
        if opens and closes and closes[0] > opens[0]:
            region = range(opens[0] + 1, closes[0])
            if closes[-1] < len(lines):
                markup = [(closes[-1] + 1, len(lines))]
            if opens[0] > 1:
                markup.insert(0, (1, opens[0] - 1))
        else:
            region, markup = range(0), [(1, len(lines))]
    starts = []
    for n in region:
        line = lines[n - 1]
        stripped = line[1:] if lang == "svelte" and line.startswith("\t") else line
        stripped = stripped[2:] if lang == "svelte" and stripped.startswith("  ") else stripped
        match = pattern.match(stripped)
        if match and stripped == stripped.lstrip():
            starts.append((n, match.group(1), match.group(2) or match.group(1)))
    out = []
    lo, hi = (region.start, region.stop - 1) if len(region) else (1, 0)
    if starts and starts[0][0] > lo:
        text = "\n".join(lines[lo - 1:starts[0][0] - 1])
        out.append(Part("head", path.rsplit("/", 1)[-1], lo, starts[0][0] - 1, refs=_code_refs(text),
                        said=_code_said(text)))
    for n, (first, kind, name) in enumerate(starts):
        last = (starts[n + 1][0] - 1) if n + 1 < len(starts) else hi
        while last > first and not lines[last - 1].strip():
            last -= 1
        text = "\n".join(lines[first - 1:last])
        out.append(Part(kind.rstrip("!*"), name, first, last, defines=[(name, name, kind.rstrip("!*"))],
                        refs=_code_refs(text), said=_code_said(text)))
    if not starts and len(region):
        text = "\n".join(lines[lo - 1:hi])
        out.append(Part("text", path.rsplit("/", 1)[-1], lo, hi, refs=_code_refs(text), said=_code_said(text)))
    for first, last in markup:
        text = "\n".join(lines[first - 1:last])
        out.append(Part("markup", path.rsplit("/", 1)[-1], first, last, refs=_code_refs(text),
                        said=_code_said(text)))
    return _merge_small(out)


SMALL_KINDS = {"let", "const", "var", "type", "static"}


def _merge_small(parts):
    """Runs of short declarations (a Svelte component's `export let` props, a block of constants) as one part of
    at most WINDOW lines: a one-line part is too little to find, and too many to rank."""
    out = []
    for part in parts:
        last = out[-1] if out else None
        small = part.kind in SMALL_KINDS and part.last - part.first < 3
        if small and last is not None and last.kind == "decls" and part.last - last.first < WINDOW \
                and part.first - last.last <= 3:
            last.last = part.last
            last.defines.extend(part.defines)
            last.refs.extend(r for r in part.refs if r not in last.refs)
            last.said.extend(t for t in part.said if t not in last.said)
            last.symbol = f"{last.symbol.split(' +')[0]} +{len(last.defines) - 1}"
        elif small:
            out.append(Part("decls", part.symbol, part.first, part.last, defines=list(part.defines),
                            refs=list(part.refs), said=list(part.said)))
        else:
            out.append(part)
    return out


def cut(path, lines):
    """The parts of a file: by its syntax, then long units cut into windows, empty ones dropped."""
    lang = language(path)
    units = None
    if lang == "python":
        try:
            units = python_parts(path, lines)
        except RecursionError:
            units = None        # too deep for the parser: cut as plain text below
    elif lang == "markdown":
        units = markdown_parts(path, lines)
    elif lang == "yaml":
        units = yaml_parts(path, lines)
    elif lang in ("ts", "svelte", "rust"):
        units = brace_parts(path, lines, lang)
    if not units:
        units = [Part("text", path.rsplit("/", 1)[-1], 1, max(1, len(lines)))]
    out = []
    for unit in _merge_short(sorted(units, key=lambda u: (u.first, u.last))):
        if unit.last < unit.first:
            continue
        pieces = windows(unit.first, unit.last) if unit.last - unit.first + 1 > MAX_LINES else [(unit.first,
                                                                                                unit.last)]
        for n, (a, b) in enumerate(pieces):
            if not any(line.strip() for line in lines[a - 1:b]):
                continue
            # A window carries the messages its own lines hold; the first one the unit's definitions and names.
            here = "\n".join(lines[a - 1:b])
            said = unit.said if len(pieces) == 1 else [t for t in unit.said if t.split("{}")[0][:40] in here]
            out.append(Part(unit.kind if len(pieces) == 1 else f"{unit.kind}~{n + 1}", unit.symbol, a, b,
                            defines=unit.defines if n == 0 else [], refs=unit.refs if n == 0 else [], said=said))
    return out


def _merge_short(units):
    """Runs of short units (under MIN_LINES each, at most 3 lines apart) as one part of at most WINDOW lines, its
    symbol the first one's with how many more it holds; a long unit is never merged."""
    out = []
    for unit in units:
        size = unit.last - unit.first + 1
        last = out[-1] if out else None
        # a component's markup and its script never share a part
        joinable = (last is not None and getattr(last, "_short", False) and size < MIN_LINES
                    and unit.first - last.last <= 3 and unit.last - last.first + 1 <= WINDOW
                    and (unit.kind == "markup") == getattr(last, "_markup", False))
        if joinable:
            last.last = unit.last
            last.defines = last.defines + [d for d in unit.defines if d not in last.defines]
            last.refs = last.refs + [r for r in unit.refs if r not in last.refs]
            last.said = last.said + [t for t in unit.said if t not in last.said]
            last._count += 1
            last.kind = "group"
            last.symbol = f"{last._first_symbol} +{last._count - 1}"
        else:
            if size < MIN_LINES:
                unit = Part(unit.kind, unit.symbol, unit.first, unit.last, defines=list(unit.defines),
                            refs=list(unit.refs), said=list(unit.said))
                unit._short, unit._count, unit._first_symbol = True, 1, unit.symbol
                unit._markup = unit.kind == "markup"
            out.append(unit)
    return out


def module_doc(path, lines):
    """The first sentence a file opens on (at most 200 characters): a Python module's docstring; the comment a YAML,
    TS, Svelte or Rust file opens with; else ''. What the map says of a file."""
    if language(path) != "python":
        said = []
        for line in lines[:30]:
            text = line.strip()
            if not text:
                if said:
                    break
                continue
            if text.startswith(("#!", "<script", "<style")):
                continue
            marker = next((m for m in ("<!--", "///", "//!", "//", "/**", "/*", "*", "#") if text.startswith(m)), None)
            if marker is None:
                break
            words = text[len(marker):].strip().rstrip("*/").rstrip("-->").strip(" =-")
            if words:
                said.append(words)
        doc = " ".join(said)
    else:
        try:
            doc = ast.get_docstring(ast.parse("\n".join(lines))) or ""
        except (SyntaxError, ValueError, RecursionError):
            return ""
    first = re.split(r"(?<=[.!?])\s", " ".join(doc.split()), maxsplit=1)[0]
    return first[:200]
