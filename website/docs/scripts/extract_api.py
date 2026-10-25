#!/usr/bin/env python3
"""Extract the Alkahest Python API surface *statically* into JSON.

Nothing here imports ``alkahest``: CI builds the docs without compiling the
extension. Three sources are combined:

* ``python/alkahest/**/*.py`` parsed with :mod:`ast` (``__all__``, import
  chains, ``def``/``class`` signatures and docstrings);
* ``alkahest-py/src/*.rs`` scanned for PyO3 items (``#[pyfunction]``,
  ``#[pyclass]``, ``#[pymethods]``) — their ``///`` comments are exactly the
  ``__doc__`` a user sees in ``help()``;
* ``docs/sphinx/api/*.rst`` — the hand-written reference, used for page /
  section grouping and as a fallback body when a symbol has no docstring.

Usage: ``python3 extract_api.py <repo-root> <out.json>``
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

GITHUB = "https://github.com/alkahest-cas/alkahest/blob/main/"

# ---------------------------------------------------------------------------
# Rust (PyO3) scanning
# ---------------------------------------------------------------------------

_STR_RE = re.compile(r'r(#*)"(?:.|\n)*?"\1|"(?:\\.|[^"\\])*"')
_CHAR_RE = re.compile(r"'(?:\\.[^']{0,8}|[^\\'\n])'")


def _strip_rust_literals(line: str) -> str:
    line = _STR_RE.sub('""', line)
    line = _CHAR_RE.sub("' '", line)
    i = line.find("//")
    return line if i < 0 else line[:i]


@dataclass
class RustItem:
    pyname: str
    kind: str  # function | class | method | property | staticmethod | classmethod | new
    doc: str
    signature: str | None
    path: str
    line: int
    members: list = field(default_factory=list)


def _attr_value(attrs: list[str], key: str) -> str | None:
    for a in attrs:
        m = re.search(key + r'\s*=\s*"((?:\\.|[^"\\])*)"', a)
        if m:
            return m.group(1).replace('\\"', '"')
    return None


def _attr_paren(attrs: list[str], key: str) -> str | None:
    """Value of ``key = ( ... )`` inside a ``#[pyo3(...)]`` attribute."""
    for a in attrs:
        m = re.search(key + r"\s*=\s*\(", a)
        if not m:
            continue
        depth, i = 1, m.end()
        while i < len(a) and depth:
            if a[i] == "(":
                depth += 1
            elif a[i] == ")":
                depth -= 1
            i += 1
        return a[m.end() : i - 1]
    return None


_SKIP_PARAM = re.compile(r"^(?:py|_py|slf|_slf|self|cls|_cls)$")


def _rust_params(sig_text: str) -> list[str]:
    """Names of the Python-visible parameters of a Rust fn signature."""
    depth, cur, parts = 0, "", []
    for ch in sig_text:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    out = []
    for p in parts:
        p = p.strip()
        if not p or p in ("&self", "&mut self", "self", "mut self"):
            continue
        if ":" not in p:
            continue
        name, ty = p.split(":", 1)
        name = name.replace("mut ", "").strip()
        ty = ty.strip()
        if _SKIP_PARAM.match(name) or ty.startswith("Python<") or "Bound<'_, PyType>" in ty:
            continue
        if name.startswith("_"):
            name = name[1:]
        out.append((name, ty.startswith("Option<")))
    # PyO3 gives trailing `Option<T>` parameters a default of None.
    names = []
    trailing = True
    for name, opt in reversed(out):
        trailing = trailing and opt
        names.append(f"{name}=None" if trailing else name)
    return list(reversed(names))


_LITERAL_DEFAULT = re.compile(r'^(?:-?[\d.]+(?:e-?\d+)?|"[^"]*"|None|True|False|\[\]|\(\)|\{\})$')


def _clean_pyo3_sig(sig: str) -> str:
    """A ``#[pyo3(signature = ...)]`` body rendered the way PyO3 renders
    ``__text_signature__``: Python literals kept, anything else ``...``."""
    sig = " ".join(sig.split()).strip().rstrip(",")
    parts, depth, cur = [], 0, ""
    for ch in sig:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "=" in p:
            name, val = (x.strip() for x in p.split("=", 1))
            val = {"true": "True", "false": "False"}.get(val, val)
            if not _LITERAL_DEFAULT.match(val):
                val = "..."
            p = f"{name}={val}"
        out.append(p.replace("r#", ""))
    return ", ".join(out)


def scan_rust(src_dir: Path, repo: Path) -> tuple[dict, dict, dict]:
    """Return (functions, classes, struct_to_pyname)."""
    functions: dict[str, RustItem] = {}
    classes: dict[str, RustItem] = {}
    struct_to_py: dict[str, str] = {}
    exceptions: dict[str, str] = {}
    pending_methods: dict[str, list[RustItem]] = {}

    for path in sorted(src_dir.rglob("*.rs")):
        rel = str(path.relative_to(repo))
        lines = path.read_text(encoding="utf-8").splitlines()
        docs: list[str] = []
        attrs: list[str] = []
        attr_buf: str | None = None
        attr_depth = 0
        depth = 0
        impl_stack: list[tuple[str, int]] = []  # (struct, depth at which impl body lives)
        pymethods_next = False
        i = 0
        while i < len(lines):
            raw = lines[i]
            s = raw.strip()
            lineno = i + 1
            i += 1
            if attr_buf is not None:
                attr_buf += " " + s
                attr_depth += s.count("[") - s.count("]")
                if attr_depth <= 0:
                    attrs.append(attr_buf)
                    attr_buf = None
                continue
            if s.startswith("///"):
                d = s[3:]
                docs.append(d[1:] if d.startswith(" ") else d)
                continue
            if s.startswith("#[") or s.startswith("#!["):
                d = s.count("[") - s.count("]")
                if d > 0:
                    attr_buf, attr_depth = s, d
                else:
                    attrs.append(s)
                continue
            if s.startswith("//") or not s:
                if not s:
                    pass
                continue

            code = _strip_rust_literals(raw)
            m_exc = re.search(
                r"create_exception!\(\s*\w+\s*,\s*(\w+)\s*,\s*[\w:]+\s*(?:,\s*\"((?:\\.|[^\"\\])*)\")?",
                raw,
            )
            if m_exc:
                exceptions[m_exc.group(1)] = m_exc.group(2) or ""
            m_add = re.search(r'\.add\(\s*"(\w+)"\s*,.*get_type(?:_bound)?::<(\w+)>', raw)
            if m_add:
                struct_to_py.setdefault(m_add.group(2), m_add.group(1))

            in_methods = bool(impl_stack) and impl_stack[-1][1] == depth and impl_stack[-1][0] != ""
            is_pyclass = any(a.startswith("#[pyclass") for a in attrs)
            is_pyfn = any(a.startswith("#[pyfunction") for a in attrs)
            is_pymethods = any(a.startswith("#[pymethods") for a in attrs)

            m_struct = re.match(r"(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum)\s+(\w+)", s)
            m_fn = re.match(
                r"(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)",
                s,
            )
            m_impl = re.match(r"impl(?:<[^>]*>)?\s+(?:[\w:]+\s+for\s+)?(\w+)", s)

            if m_struct and is_pyclass:
                sname = m_struct.group(1)
                pyname = _attr_value(attrs, "name") or sname
                struct_to_py[sname] = pyname
                item = RustItem(pyname, "class", "\n".join(docs).strip(), None, rel, lineno)
                classes.setdefault(sname, item)
                # #[pyo3(get)] fields
                if "{" in code and not code.rstrip().endswith("}"):
                    fdocs: list[str] = []
                    fattrs: list[str] = []
                    while i < len(lines):
                        fl = lines[i].strip()
                        i += 1
                        if fl.startswith("}"):
                            break
                        if fl.startswith("///"):
                            d = fl[3:]
                            fdocs.append(d[1:] if d.startswith(" ") else d)
                            continue
                        if fl.startswith("#["):
                            fattrs.append(fl)
                            continue
                        mf = re.match(r"(?:pub(?:\([^)]*\))?\s+)?(\w+)\s*:", fl)
                        if mf and any("get" in a for a in fattrs if a.startswith("#[pyo3")):
                            fname = _attr_value(fattrs, "name") or mf.group(1)
                            item.members.append(
                                RustItem(fname, "property", "\n".join(fdocs).strip(), None, rel, i)
                            )
                        if mf or fl.endswith(","):
                            fdocs, fattrs = [], []
                docs, attrs = [], []
                continue

            if m_impl and not m_fn:
                if is_pymethods or pymethods_next:
                    impl_stack.append((m_impl.group(1), depth + code.count("{") - code.count("}")))
                else:
                    impl_stack.append(("", depth + code.count("{") - code.count("}")))
                depth += code.count("{") - code.count("}")
                docs, attrs = [], []
                continue

            if m_fn and (is_pyfn or in_methods):
                rname = m_fn.group(1)
                # gather the full parameter list
                sig_lines = [code]
                j = i
                while sig_lines[-1].count("(") == 0 or (
                    "".join(sig_lines).count("(") > "".join(sig_lines).count(")")
                ):
                    if j >= len(lines):
                        break
                    sig_lines.append(_strip_rust_literals(lines[j]))
                    j += 1
                joined = " ".join(sig_lines)
                a = joined.find("(")
                k, d2 = a + 1, 1
                while k < len(joined) and d2:
                    if joined[k] == "(":
                        d2 += 1
                    elif joined[k] == ")":
                        d2 -= 1
                    k += 1
                params = joined[a + 1 : k - 1]
                pyname = _attr_value(attrs, "name")
                kind = "function"
                if in_methods:
                    kind = "method"
                    if any(x.startswith("#[new") for x in attrs):
                        kind = "new"
                    elif any(x.startswith("#[getter") for x in attrs):
                        kind = "property"
                        mg = re.match(
                            r"#\[getter\((\w+)\)\]",
                            next(x for x in attrs if x.startswith("#[getter")),
                        )
                        if mg:
                            pyname = mg.group(1)
                        elif not pyname:
                            pyname = rname[4:] if rname.startswith("get_") else rname
                    elif any(x.startswith("#[setter") for x in attrs):
                        kind = "setter"
                    elif any(x.startswith("#[staticmethod") for x in attrs):
                        kind = "staticmethod"
                    elif any(x.startswith("#[classmethod") for x in attrs):
                        kind = "classmethod"
                pyname = pyname or rname
                text_sig = _attr_value(attrs, "text_signature")
                if text_sig:
                    sig = text_sig.strip()
                    sig = re.sub(r"^\(\s*\$?(?:self|cls)\s*,?\s*", "(", sig)
                else:
                    psig = _attr_paren(attrs, "signature")
                    if psig is not None:
                        sig = "(" + _clean_pyo3_sig(psig) + ")"
                    else:
                        sig = "(" + ", ".join(_rust_params(params)) + ")"
                item = RustItem(pyname, kind, "\n".join(docs).strip(), sig, rel, lineno)
                if in_methods:
                    if kind != "setter":
                        pending_methods.setdefault(impl_stack[-1][0], []).append(item)
                elif not any("cfg(test" in x for x in attrs):
                    functions.setdefault(pyname, item)
                docs, attrs = [], []
                # fall through to brace accounting for the fn line
            elif not s.startswith("#"):
                docs, attrs = [], []

            pymethods_next = False
            delta = code.count("{") - code.count("}")
            depth += delta
            while impl_stack and depth < impl_stack[-1][1]:
                impl_stack.pop()

    for sname, items in pending_methods.items():
        cls = classes.get(sname)
        if cls is None:
            continue
        for it in items:
            if it.kind == "new":
                cls.signature = it.signature
                if not cls.doc and it.doc:
                    cls.doc = it.doc
                continue
            cls.members.append(it)
    by_py = {}
    for sname, item in classes.items():
        by_py.setdefault(struct_to_py.get(sname, item.pyname), item)
    exc_by_py = {struct_to_py.get(k, k): v for k, v in exceptions.items()}
    return functions, by_py, exc_by_py


# ---------------------------------------------------------------------------
# Python (ast) scanning
# ---------------------------------------------------------------------------


@dataclass
class Binding:
    kind: str  # def | class | import | module | assign | native
    node: object = None
    module: str = ""  # for import: source module (dotted, relative to alkahest)
    name: str = ""  # for import: original name
    file: str = ""


class PyModule:
    def __init__(self, dotted: str, path: Path, repo: Path):
        self.dotted = dotted
        self.path = path
        self.rel = str(path.relative_to(repo))
        self.tree = ast.parse(path.read_text(encoding="utf-8"))
        self.doc = ast.get_docstring(self.tree) or ""
        self.bindings: dict[str, Binding] = {}
        self.all: list[str] | None = None
        self.defined_order: list[str] = []
        self._walk(self.tree.body, guarded=False)

    def _resolve_rel(self, level: int, module: str | None) -> str:
        parts = self.dotted.split(".")
        if self.path.name != "__init__.py":
            parts = parts[:-1]
        if level > 1:
            parts = parts[: len(parts) - (level - 1)]
        base = ".".join(parts)
        return f"{base}.{module}" if module else base

    def _bind(self, name: str, b: Binding, guarded: bool):
        if guarded and name in self.bindings:
            return
        if name not in self.bindings:
            self.defined_order.append(name)
        self.bindings[name] = b

    def _walk(self, body, guarded: bool):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._bind(node.name, Binding("def", node, file=self.rel), guarded)
            elif isinstance(node, ast.ClassDef):
                self._bind(node.name, Binding("class", node, file=self.rel), guarded)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    continue
                src = self._resolve_rel(node.level, node.module)
                for a in node.names:
                    nm = a.asname or a.name
                    self._bind(nm, Binding("import", module=src, name=a.name), guarded)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        if t.id == "__all__":
                            with contextlib.suppress(Exception):
                                self.all = list(ast.literal_eval(node.value))
                        else:
                            self._bind(t.id, Binding("assign", node, file=self.rel), guarded)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                self._bind(node.target.id, Binding("assign", node, file=self.rel), guarded)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                    try:
                        extra = list(ast.literal_eval(node.value))
                    except Exception:
                        extra = []
                    self.all = (self.all or []) + extra
            elif isinstance(node, ast.If):
                # `if "solve" not in dir(): def solve(...)` style fallbacks
                # never shadow an earlier binding.
                self._walk(node.body, True)
                self._walk(node.orelse, True)
            elif isinstance(node, ast.Try):
                self._walk(node.body, guarded)
                self._walk(node.orelse, guarded)
                for h in node.handlers:
                    self._walk(h.body, True)
            elif isinstance(node, ast.With):
                self._walk(node.body, guarded)


def _fmt_args(args: ast.arguments, drop_first: bool) -> str:
    a = ast.arguments(
        posonlyargs=list(args.posonlyargs),
        args=list(args.args),
        vararg=args.vararg,
        kwonlyargs=args.kwonlyargs,
        kw_defaults=args.kw_defaults,
        kwarg=args.kwarg,
        defaults=list(args.defaults),
    )
    if drop_first:
        if a.posonlyargs:
            a.posonlyargs.pop(0)
        elif a.args:
            a.args.pop(0)
    return ast.unparse(a)


def _py_sig(node, drop_first=False) -> str:
    s = "(" + _fmt_args(node.args, drop_first) + ")"
    if node.returns is not None:
        s += " -> " + ast.unparse(node.returns)
    return s


def _decorators(node) -> list[str]:
    out = []
    for d in node.decorator_list:
        with contextlib.suppress(Exception):
            out.append(ast.unparse(d))
    return out


# ---------------------------------------------------------------------------
# Docstring → structured markdown
# ---------------------------------------------------------------------------

_ROLE_RE = re.compile(
    r":(?:py:)?(func|class|exc|meth|attr|mod|data|obj|const|ref|doc|term|math|code|kbd):`([^`]+)`"
)
_NUMPY_SECTIONS = {
    "parameters": "Parameters",
    "params": "Parameters",
    "other parameters": "Other parameters",
    "keyword arguments": "Keyword arguments",
    "returns": "Returns",
    "return": "Returns",
    "yields": "Yields",
    "raises": "Raises",
    "raise": "Raises",
    "warns": "Warns",
    "attributes": "Attributes",
    "methods": "Methods",
    "examples": "Examples",
    "example": "Examples",
    "notes": "Notes",
    "note": "Notes",
    "see also": "See also",
    "references": "References",
    "warnings": "Warnings",
    "warning": "Warnings",
}
_GOOGLE_SECTIONS = {
    "args": "Parameters",
    "arguments": "Parameters",
    "parameters": "Parameters",
    "params": "Parameters",
    "keyword args": "Keyword arguments",
    "kwargs": "Keyword arguments",
    "returns": "Returns",
    "return": "Returns",
    "yields": "Yields",
    "raises": "Raises",
    "attributes": "Attributes",
    "example": "Examples",
    "examples": "Examples",
    "note": "Notes",
    "notes": "Notes",
    "see also": "See also",
}
_PARAM_SECTIONS = {
    "Parameters",
    "Other parameters",
    "Keyword arguments",
    "Returns",
    "Yields",
    "Raises",
    "Warns",
    "Attributes",
}


class Linker:
    """Resolves ``:func:`name``` style references to page URLs."""

    def __init__(self):
        self.urls: dict[str, str] = {}

    def resolve(self, target: str) -> str | None:
        t = target.strip().lstrip("~").rstrip("()")
        for cand in (t, t.removeprefix("alkahest."), t.removeprefix("alkahest.alkahest.")):
            if cand in self.urls:
                return self.urls[cand]
        tail = t.split(".")[-1]
        if tail in self.urls:
            return self.urls[tail]
        return None


LINKER = Linker()


def _inline(text: str) -> str:
    """RST inline markup → markdown."""

    def role(m):
        r, body = m.group(1), m.group(2)
        if r == "math":
            return f'<span class="kx">{_html_escape(body)}</span>'
        label = body
        target = body
        mm = re.match(r"(.*?)\s*<([^>]+)>$", body)
        if mm:
            label, target = mm.group(1), mm.group(2)
        if label.startswith("~"):
            label = label[1:].split(".")[-1]
        label = label.removeprefix("alkahest.")
        if r in ("func", "meth") and not label.endswith(")"):
            label += "()"
        url = LINKER.resolve(target) if r not in ("ref", "doc", "term", "code", "kbd") else None
        code = f"`{label}`"
        return f"[{code}]({url})" if url else code

    out = []
    # protect inline literals ``...`` first
    parts = re.split(r"(``.+?``)", text)
    for p in parts:
        if p.startswith("``") and p.endswith("``") and len(p) > 4:
            inner = p[2:-2]
            tick = "``" if "`" in inner else "`"
            pad = " " if tick == "``" else ""
            out.append(f"{tick}{pad}{inner}{pad}{tick}")
            continue
        p = _ROLE_RE.sub(role, p)
        p = re.sub(r"`([^`<]+?)\s*<(https?://[^>]+)>`__?", r"[\1](\2)", p)
        p = re.sub(r"\\ ", "", p)
        out.append(p)
    return "".join(out)


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _guess_lang(code: str) -> str:
    if re.search(r"^\s*(double|void|static|#include)\b", code, re.M):
        return "c"
    if re.search(r"^\s*(theorem|example|lemma|import Mathlib)\b", code, re.M):
        return "lean"
    if re.search(r"^\s*(fn |let |use |impl )", code, re.M):
        return "rust"
    if re.search(r"^\s*\$ |^\s*(pip|cargo|maturin) ", code, re.M):
        return "bash"
    if re.search(r"^\s*(module|func\.func|%)", code, re.M) and "stablehlo" in code:
        return "text"
    return "python"


def _dedent(lines: list[str]) -> list[str]:
    ind = min((len(x) - len(x.lstrip()) for x in lines if x.strip()), default=0)
    return [x[ind:] for x in lines]


def _indent_of(s: str) -> int:
    return len(s) - len(s.lstrip())


def rst_to_md(text: str) -> str:
    """Convert an RST-flavoured docstring body (no section headers) to markdown."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    prev_literal = False  # previous paragraph ended with '::'
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            out.append("")
            i += 1
            continue
        # doctest block
        if s.startswith(">>>"):
            block = []
            base = _indent_of(line)
            while i < len(lines) and lines[i].strip():
                block.append(lines[i][base:] if _indent_of(lines[i]) >= base else lines[i].strip())
                i += 1
            out += ["```python", *block, "```", ""]
            continue
        # directives
        m = re.match(r"\.\.\s+([\w-]+)::\s*(.*)$", s)
        if m:
            name, arg = m.group(1), m.group(2)
            base = _indent_of(line)
            i += 1
            body = []
            while i < len(lines) and (not lines[i].strip() or _indent_of(lines[i]) > base):
                body.append(lines[i])
                i += 1
            while body and not body[-1].strip():
                body.pop()
            body = _dedent(body)
            # strip directive options
            while body and re.match(r"\s*:\w[\w-]*:", body[0]):
                body.pop(0)
            if name in ("code-block", "code", "sourcecode"):
                lang = arg or "text"
                out += [f"```{lang}", *body, "```", ""]
            elif name == "math":
                tex = "\n".join(([arg] if arg else []) + body).strip()
                out += [f'<div class="kx-display">{_html_escape(tex)}</div>', ""]
            elif name in (
                "note",
                "warning",
                "tip",
                "important",
                "caution",
                "attention",
                "hint",
                "danger",
                "seealso",
            ):
                label = {"seealso": "See also"}.get(name, name.capitalize())
                inner = rst_to_md("\n".join(([arg] if arg else []) + body))
                out.append(f"> **{label}** ")
                out += ["> " + x if x else ">" for x in inner.split("\n")]
                out.append("")
            elif name in ("versionadded", "versionchanged", "deprecated"):
                label = {
                    "versionadded": "Added in",
                    "versionchanged": "Changed in",
                    "deprecated": "Deprecated since",
                }[name]
                inner = rst_to_md("\n".join(body))
                out.append(f"> **{label} {arg}** " + inner.replace("\n", " "))
                out.append("")
            elif name in ("rubric",):
                out += [f"**{arg}**", ""]
            else:
                inner = rst_to_md("\n".join(body))
                out += [inner, ""]
            continue
        # indented block
        if out and _indent_of(line) > 0 and (prev_literal or (out and out[-1] == "")):
            base = _indent_of(line)
            block = []
            while i < len(lines) and (not lines[i].strip() or _indent_of(lines[i]) >= base):
                block.append(lines[i])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            block = _dedent(block)
            if prev_literal:
                code = "\n".join(block)
                out += [f"```{_guess_lang(code)}", code, "```", ""]
            else:
                is_list = bool(re.match(r"[-*+]\s|\d+[.)]\s", block[0].lstrip()))
                inner = rst_to_md("\n".join(block))
                out += (
                    [inner, ""]
                    if is_list
                    else ["> " + x if x else ">" for x in inner.split("\n")] + [""]
                )
            prev_literal = False
            continue
        # paragraph (gather until blank or a less-indented structural line)
        para = []
        base = _indent_of(line)
        while i < len(lines) and lines[i].strip():
            if para and re.match(r"\s*(>>>|\.\.\s+[\w-]+::)", lines[i]):
                break
            if para and _indent_of(lines[i]) > base and para[-1].rstrip().endswith("::"):
                break
            para.append(lines[i])
            i += 1
        ptxt = "\n".join(x[base:] if _indent_of(x) >= base else x.strip() for x in para)
        literal = ptxt.rstrip().endswith("::")
        if literal:
            t = ptxt.rstrip()[:-2].rstrip()
            ptxt = (t + ":") if t and not t.endswith(":") else t
        # field list → keep (handled by section parser when relevant)
        ptxt = re.sub(r"^(\s*)\* ", r"\1- ", ptxt, flags=re.M)
        if ptxt:
            out.append(_inline(ptxt))
        prev_literal = literal
        if literal and i < len(lines) and not lines[i].strip():
            # keep prev_literal across the single blank line
            out.append("")
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and _indent_of(lines[i]) > base:
                blk_base = _indent_of(lines[i])
                block = []
                while i < len(lines) and (not lines[i].strip() or _indent_of(lines[i]) >= blk_base):
                    block.append(lines[i])
                    i += 1
                while block and not block[-1].strip():
                    block.pop()
                block = _dedent(block)
                code = "\n".join(block)
                out += [f"```{_guess_lang(code)}", code, "```", ""]
            prev_literal = False
    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _parse_param_entries(lines: list[str], section: str) -> list[dict]:
    """NumPy-style ``name : type`` + indented description entries."""
    lines = _dedent(lines)
    entries: list[dict] = []
    cur = None
    for ln in lines:
        if ln.strip() and _indent_of(ln) == 0:
            head = ln.strip()
            m = re.match(r"^([*\w][\w, *.\[\]|]*?)\s*:\s*(.*)$", head)
            if section in ("Returns", "Yields", "Raises", "Warns") and not m:
                cur = {"name": "", "type": head, "desc": []}
            elif m:
                cur = {"name": m.group(1).strip(), "type": m.group(2).strip(), "desc": []}
            else:
                cur = {"name": head, "type": "", "desc": []}
            if section in ("Returns", "Yields") and m and not re.match(r"^\w+$", m.group(1)):
                cur = {"name": "", "type": head, "desc": []}
            if section in ("Returns", "Yields") and head.endswith(".") and len(head.split()) > 4:
                cur = {"name": "", "type": "", "desc": [head]}
            entries.append(cur)
        elif cur is not None:
            cur["desc"].append(ln)
        elif ln.strip():
            entries.append({"name": "", "type": "", "desc": [ln]})
            cur = entries[-1]
    out = []
    for e in entries:
        desc = "\n".join(_dedent(e["desc"])).strip()
        if section in ("Raises", "Warns") and e["name"] and not e["type"]:
            e["type"], e["name"] = e["name"], ""
        out.append(
            {
                "name": e["name"],
                "type": _inline(e["type"]) if e["type"] else "",
                "desc": rst_to_md(desc),
            }
        )
    return out


def _parse_google_entries(lines: list[str], section: str) -> list[dict]:
    lines = _dedent(lines)
    entries = []
    cur = None
    for ln in lines:
        if ln.strip() and _indent_of(ln) == 0:
            m = re.match(r"^([*\w.]+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$", ln.strip())
            if m and section not in ("Returns", "Yields"):
                cur = {"name": m.group(1), "type": m.group(2) or "", "desc": [m.group(3)]}
            elif m and section in ("Returns", "Yields") and m.group(1)[0].isupper():
                cur = {"name": "", "type": m.group(1), "desc": [m.group(3)]}
            else:
                if cur is None or section in ("Returns", "Yields"):
                    cur = {"name": "", "type": "", "desc": [ln.strip()]}
                else:
                    cur["desc"].append(ln)
                    continue
            entries.append(cur)
        elif cur is not None:
            cur["desc"].append(ln.strip())
    out = []
    for e in entries:
        if section == "Raises" and e["name"] and not e["type"]:
            e["type"], e["name"] = e["name"], ""
        out.append(
            {
                "name": e["name"],
                "type": _inline(e["type"]) if e["type"] else "",
                "desc": rst_to_md("\n".join(e["desc"]).strip()),
            }
        )
    return out


def _field_list(lines: list[str]) -> tuple[list[str], dict[str, list[dict]]]:
    """Pull ``:param x:`` / ``:returns:`` / ``:raises X:`` fields out of a body."""
    rest, fields = [], {}
    types: dict[str, str] = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(
            r"^(\s*):(param|parameter|arg|argument|key|keyword|type|returns?|rtype|raises?|except|yields?)(?:\s+([^:]+))?:\s*(.*)$",
            ln,
        )
        if not m:
            rest.append(ln)
            i += 1
            continue
        base = len(m.group(1))
        kind, arg, txt = m.group(2), (m.group(3) or "").strip(), m.group(4)
        body = [txt]
        i += 1
        while i < len(lines) and lines[i].strip() and _indent_of(lines[i]) > base:
            body.append(lines[i].strip())
            i += 1
        desc = " ".join(body).strip()
        if kind == "type":
            types[arg] = desc
            continue
        if kind == "rtype":
            fields.setdefault("Returns", [{"name": "", "type": "", "desc": ""}])[0]["type"] = (
                _inline(desc)
            )
            continue
        sec = {
            "param": "Parameters",
            "parameter": "Parameters",
            "arg": "Parameters",
            "argument": "Parameters",
            "key": "Parameters",
            "keyword": "Parameters",
            "returns": "Returns",
            "return": "Returns",
            "raises": "Raises",
            "raise": "Raises",
            "except": "Raises",
            "yields": "Yields",
            "yield": "Yields",
        }[kind]
        name, typ = arg, ""
        if sec == "Parameters" and " " in arg:
            typ, name = arg.rsplit(" ", 1)
        if sec == "Raises":
            name, typ = "", arg
        if sec == "Returns" and fields.get("Returns"):
            fields["Returns"][0]["desc"] = rst_to_md(desc)
            continue
        fields.setdefault(sec, []).append(
            {"name": name, "type": _inline(typ) if typ else "", "desc": rst_to_md(desc)}
        )
    for e in fields.get("Parameters", []):
        if not e["type"] and e["name"] in types:
            e["type"] = _inline(types[e["name"]])
    return rest, fields


def _split_trailing_prose(body: list[str]) -> tuple[list[str], list[str]]:
    """A NumPy/Google field section ends where an unindented paragraph that
    reads as prose (not ``name : type``) follows a blank line."""
    lines = _dedent(body)
    for i in range(1, len(lines)):
        ln = lines[i]
        if not ln.strip() or _indent_of(ln) or lines[i - 1].strip():
            continue
        s = ln.strip()
        header_like = (
            bool(re.match(r"^[*\w][\w.]*(\s*,\s*[*\w][\w.]*)*\s*(:\s*.*)?$", s))
            and len(s.split(":")[0].split()) <= 3
        )
        prose = (
            "``" in s or ":func:" in s or ":class:" in s or s.endswith(".") or len(s.split()) > 6
        )
        if prose and not header_like:
            return lines[:i], lines[i:]
    return lines, []


def parse_docstring(doc: str) -> list[dict]:
    """Docstring → list of blocks ``{"t": "md", "md": str}`` /
    ``{"t": "params", "title": str, "items": [...]}``."""
    if not doc or not doc.strip():
        return []
    lines = doc.expandtabs(4).split("\n")
    # split into (title, lines) sections
    sections: list[tuple[str | None, str, list[str]]] = [(None, "", [])]
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        key = s.lower().rstrip(":")
        if s and nxt and set(nxt) <= {"-", "="} and len(nxt) >= 3 and abs(len(nxt) - len(s)) <= 3:
            # NumPy-style header; unknown titles ("Verification", "Trust")
            # become plain titled sections
            sections.append((_NUMPY_SECTIONS.get(key, s.rstrip(":")), "numpy", []))
            i += 2
            continue
        # a Google-style header only when followed by an indented line
        if (
            s.endswith(":")
            and key in _GOOGLE_SECTIONS
            and _indent_of(ln) == 0
            and s[0].isupper()
            and i + 1 < len(lines)
            and lines[i + 1].strip()
            and _indent_of(lines[i + 1]) > 0
        ):
            sections.append((_GOOGLE_SECTIONS[key], "google", []))
            i += 1
            continue
        sections[-1][2].append(ln)
        i += 1
    blocks: list[dict] = []
    for title, style, body in sections:
        while body and not body[-1].strip():
            body.pop()
        while body and not body[0].strip():
            body.pop(0)
        if not body:
            continue
        if title is None:
            rest, fields = _field_list(body)
            md = rst_to_md("\n".join(rest))
            if md:
                blocks.append({"t": "md", "md": md})
            for sec in ("Parameters", "Returns", "Yields", "Raises"):
                if sec in fields:
                    blocks.append({"t": "params", "title": sec, "items": fields[sec]})
            continue
        if title in _PARAM_SECTIONS:
            body, trailing = _split_trailing_prose(body)
            items = (_parse_param_entries if style == "numpy" else _parse_google_entries)(
                body, title
            )
            if items:
                # merge into an existing block of the same title
                for b in blocks:
                    if b["t"] == "params" and b["title"] == title:
                        b["items"] += items
                        break
                else:
                    blocks.append({"t": "params", "title": title, "items": items})
            if trailing:
                md = rst_to_md("\n".join(_dedent(trailing)))
                if md:
                    blocks.append({"t": "md", "md": md})
            continue
        if title == "See also":
            md = rst_to_md("\n".join(_dedent(body)))
            # NumPy "name : desc" lines → links
            seen = []
            for ln in _dedent(body):
                m = re.match(r"^([\w.]+)(?:\s*:\s*(.*))?$", ln.strip())
                if m and ln.strip():
                    seen.append(m)
            if seen and len(seen) == len(
                [x for x in body if x.strip() and _indent_of(x) == _indent_of(body[0])]
            ):
                parts = []
                for m in seen:
                    url = LINKER.resolve(m.group(1))
                    lbl = f"`{m.group(1)}`"
                    parts.append(
                        f"- {'[' + lbl + '](' + url + ')' if url else lbl}"
                        + (f" — {_inline(m.group(2))}" if m.group(2) else "")
                    )
                md = "\n".join(parts)
            blocks.append({"t": "md", "md": f"**See also**\n\n{md}"})
            continue
        md = rst_to_md("\n".join(_dedent(body)))
        if title == "Examples":
            blocks.append({"t": "md", "md": md, "title": "Examples"})
        else:
            blocks.append({"t": "md", "md": md, "title": title})
    return blocks


def summary_of(doc: str) -> str:
    if not doc:
        return ""
    para = []
    for ln in doc.strip().split("\n"):
        if not ln.strip():
            break
        para.append(ln.strip())
    s = " ".join(para)
    s = re.sub(r"``([^`]+)``", r"\1", s)
    s = _ROLE_RE.sub(
        lambda m: (
            m.group(2).lstrip("~").split(".")[-1]
            if "<" not in m.group(2)
            else m.group(2).split("<")[0].strip()
        ),
        s,
    )
    s = s.replace("`", "").replace("**", "").replace("\\ ", "")
    s = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"\1", s)
    m = re.match(r"(.+?[.!?])(\s|$)", s)
    out = m.group(1) if m else s
    return out[:240]


# ---------------------------------------------------------------------------
# Sphinx RST reference pages (grouping + fallback prose)
# ---------------------------------------------------------------------------

_DIRECTIVE = re.compile(
    r"^(\s*)\.\.\s+(function|class|exception|data|method|classmethod|staticmethod|attribute|property)::\s*(.+)$"
)


def parse_rst_page(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").split("\n")
    title = ""
    i = 0
    # title
    while i < len(lines) - 1:
        if lines[i].strip() and set(lines[i + 1].strip()) == {"="}:
            title = lines[i].strip()
            i += 2
            break
        i += 1
    sections = [{"title": None, "items": []}]

    def add_prose(buf):
        txt = "\n".join(buf).strip("\n")
        if txt.strip():
            sections[-1]["items"].append({"t": "prose", "rst": txt})

    buf: list[str] = []
    while i < len(lines):
        ln = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if (
            ln.strip()
            and nxt.strip()
            and set(nxt.strip()) <= {"-", "~", "^"}
            and len(nxt.strip()) >= 3
            and _indent_of(ln) == 0
        ):
            add_prose(buf)
            buf = []
            sections.append({"title": ln.strip(), "items": []})
            i += 2
            continue
        if re.match(r"^\.\.\s+currentmodule::", ln):
            i += 1
            continue
        m = _DIRECTIVE.match(ln)
        if m and _indent_of(ln) == 0:
            add_prose(buf)
            buf = []
            group = []
            # consecutive directives (signature-only) form a group
            while i < len(lines):
                m = _DIRECTIVE.match(lines[i])
                if not (m and _indent_of(lines[i]) == 0):
                    break
                group.append((m.group(2), m.group(3).strip()))
                i += 1
            body = []
            while i < len(lines) and (not lines[i].strip() or _indent_of(lines[i]) > 0):
                body.append(lines[i])
                i += 1
            while body and not body[-1].strip():
                body.pop()
            body = _dedent(body)
            # nested member directives
            members = []
            own = []
            j = 0
            while j < len(body):
                mm = _DIRECTIVE.match(body[j])
                if (
                    mm
                    and _indent_of(body[j]) == 0
                    and mm.group(2)
                    in ("method", "classmethod", "staticmethod", "attribute", "property")
                ):
                    sig = mm.group(3).strip()
                    j += 1
                    mb = []
                    while j < len(body) and (not body[j].strip() or _indent_of(body[j]) > 0):
                        mb.append(body[j])
                        j += 1
                    while mb and not mb[-1].strip():
                        mb.pop()
                    members.append({"kind": mm.group(2), "sig": sig, "rst": "\n".join(_dedent(mb))})
                else:
                    own.append(body[j])
                    j += 1
            if members:
                # bold run-in labels that only grouped the nested member
                # directives ("**Persistence**") are orphaned once members
                # are rendered separately
                own = [x for x in own if not re.match(r"^\s*\*\*[^*]+\*\*\s*$", x)]
            sections[-1]["items"].append(
                {"t": "sym", "group": group, "rst": "\n".join(own).strip("\n"), "members": members}
            )
            continue
        buf.append(ln)
        i += 1
    add_prose(buf)
    sections = [s for s in sections if s["items"] or s["title"]]
    return {"title": title, "sections": sections}


def _split_sig(sig: str) -> tuple[str, str]:
    m = re.match(r"^([\w.]+)\s*(\(.*)?$", sig)
    if not m:
        return sig, ""
    return m.group(1), (m.group(2) or "")


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

API_PAGES = [
    # (slug, sphinx file, group)
    ("core", "core", "Core"),
    ("simplify", "simplify", "Core"),
    ("diff", "diff", "Core"),
    ("poly", "poly", "Core"),
    ("numerics", "numerics", "Core"),
    ("transform", "transform", "Core"),
    ("pari", "pari", "Core"),
    ("units", "units", "Core"),
    ("tensor", "tensor", "Core"),
    ("controls", "controls", "Core"),
    ("orthopoly", "orthopoly", "Core"),
    ("noncommutative", "noncommutative", "Core"),
    ("specfun", "specfun", "Core"),
    ("matrix", "matrix", "Advanced"),
    ("ode", "ode", "Advanced"),
    ("solve", "solve", "Advanced"),
    ("linprog", "linprog", "Advanced"),
    ("logic", "logic", "Advanced"),
    ("codegen", "codegen", "Advanced"),
    ("workload", "workload", "Advanced"),
    ("autoresearch", "autoresearch", "Advanced"),
    ("certificates", None, "Advanced"),
    ("errors", "errors", "Advanced"),
]


# Public symbols the hand-written Sphinx pages never got round to. Each is
# appended to the page where a reader would look for it.
EXTRA_SECTIONS = {
    "core": [
        (
            "Assumptions and context",
            "Assumptions Domain active_assumptions active_domain get_context_value",
        ),
        (
            "Version and capabilities",
            "version __version__ capabilities HAS_EGRAPH jit_is_available",
        ),
    ],
    "simplify": [
        ("Rational functions", "apart cancel together"),
        (
            "More rewriting",
            "simplify_trig_normal_form simplify_clifford_orthogonal simplify_pauli "
            "EgraphConfig ZeroTest",
        ),
    ],
    "diff": [
        ("Limits and series", "limit series Series residue"),
        (
            "Sums, products and recurrences",
            "sum_definite sum_indefinite product_definite product_indefinite Product rsolve "
            "solve_linear_recurrence_homogeneous zeilberger ZeilbergerCertificate verify_wz_pair "
            "guess_holonomic GuessedRecurrence GUESS_STATUSES GUESS_STATUS_MEANINGS "
            "supercongruence_sweep CongruenceSweep ModularRecurrence",
        ),
    ],
    "poly": [
        (
            "Roots, resultants and factorisation",
            "real_roots refine_root RootInterval resultant subresultant_prs routh_hurwitz "
            "factor_univariate_mod_p UniPolyFactorization UniPolyFactorModP MultiPolyFactorization",
        ),
        ("Modular arithmetic", "binomial_mod"),
    ],
    "numerics": [
        (
            "Evaluation",
            "evaluate EvaluationResult ModularEvaluation guess_relation relation_confidence",
        ),
        (
            "Rigorous bounds",
            "bound_on_box bounds_supported BoundsSupport Enclosure verified_integral "
            "verified_no_roots "
            "verified_sign verified_ode VerifiedOdeSolution",
        ),
    ],
    "specfun": [
        (
            "More elementary and special functions",
            "acosh asinh atanh arg re im conjugate bessel_j0 bessel_j1 digamma trigamma dilog "
            "elliptic_k elliptic_e elliptic_f elliptic_pi exp_integral_ei log_integral "
            "sin_integral "
            "cos_integral sinh_integral cosh_integral fresnels fresnelc lambert_w lambert_wm1",
        ),
    ],
    "matrix": [("Side conditions", "matrix_exp_side_conditions matrix_inverse_side_conditions")],
    "ode": [
        (
            "Differential algebra",
            "dae_index_reduce DaeIndexReduction rosenfeld_groebner RosenfeldGroebnerResult",
        )
    ],
    "solve": [
        (
            "More solving",
            "solve_numerical solve_side_conditions CertifiedSolution triangularize RegularChain "
            "primary_decomposition PrimaryComponent radical diophantine DiophantineSolution "
            "pell_classes",
        ),
        ("Quantifier elimination", "Exists Forall decide cad_project cad_lift"),
        ("Positivity", "prove_nonneg sos_decompose PositivityCertificate"),
    ],
    "logic": [("Satisfiability", "satisfiable MinimizedLogic")],
    "codegen": [
        ("Compilation", "CompileCache CompiledGradTracedFn CudaCompiledFn cuda_device_count")
    ],
    "noncommutative": [("Lie algebras", "LieAlgebra")],
    "tensor": [("Indices", "TensorIndex")],
    "certificates": [
        ("Lean certificates", "to_lean"),
        ("Certifiability", "certifiable Certifiability require_certificate certificate_coverage"),
    ],
}

# Where symbols that no hand-written reference page mentions are listed,
# keyed by the file that defines them.
_MORE_GROUPS = [
    (
        "number-theory",
        "Number theory, lattices and modular arithmetic",
        r"(number_theory|lattice|modular|theta|bigint|numfield|character)",
    ),
    (
        "groups",
        "Groups, coding theory and finite fields",
        r"(group|fpgroup|matgroup|stabilizer|coding|ffield|funcfield)",
    ),
    ("plotting", "Plotting, printing and parsing", r"(_plot|_pretty|_parse)"),
    ("probability", "Probability and statistics", r"(joint_prob|probability)"),
    ("more", "Everything else", r".*"),
]


def main():
    repo = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2])
    pkg = repo / "python" / "alkahest"

    rust_fns, rust_classes, rust_exc = scan_rust(repo / "alkahest-py" / "src", repo)

    modules: dict[str, PyModule] = {}

    def load(dotted: str) -> PyModule | None:
        if dotted in modules:
            return modules[dotted]
        rel = dotted.split(".")[1:]
        cand = (
            [pkg.joinpath(*rel).with_suffix(".py"), pkg.joinpath(*rel, "__init__.py")]
            if rel
            else [pkg / "__init__.py"]
        )
        for c in cand:
            if c.exists():
                modules[dotted] = PyModule(dotted, c, repo)
                return modules[dotted]
        return None

    root = load("alkahest")
    assert root is not None

    symbols: dict[str, dict] = {}

    def src_link(rel: str, line: int) -> str:
        return f"{GITHUB}{rel}#L{line}"

    def native_entry(name: str, qual: str) -> dict | None:
        if name in rust_fns:
            f = rust_fns[name]
            return {
                "kind": "function",
                "signature": f.signature,
                "doc": f.doc,
                "source": src_link(f.path, f.line),
                "origin": "native",
                "_members": [],
            }
        if name in rust_classes:
            c = rust_classes[name]
            is_exc = False
            return {
                "kind": "exception" if is_exc else "class",
                "signature": c.signature,
                "doc": c.doc,
                "source": src_link(c.path, c.line),
                "origin": "native",
                "_members": c.members,
            }
        if name in rust_exc:
            return {
                "kind": "exception",
                "signature": None,
                "doc": rust_exc[name],
                "source": None,
                "origin": "native",
                "_members": [],
            }
        return None

    def py_entry(b: Binding, mod: PyModule) -> dict:
        node = b.node
        if b.kind == "def":
            return {
                "kind": "function",
                "signature": _py_sig(node),
                "doc": inspect.cleandoc(ast.get_docstring(node, clean=False) or ""),
                "source": src_link(mod.rel, node.lineno),
                "origin": "python",
                "_node": node,
            }
        if b.kind == "class":
            init = next(
                (
                    n
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == "__init__"
                ),
                None,
            )
            bases = [ast.unparse(x) for x in node.bases]
            is_exc = any(
                re.search(r"(Error|Exception|Warning)$", x) for x in bases
            ) or node.name.endswith("Error")
            sig = _py_sig(init, drop_first=True) if init else None
            if sig and sig.endswith(" -> None"):
                sig = sig[: -len(" -> None")]
            return {
                "kind": "exception" if is_exc else "class",
                "signature": sig,
                "doc": inspect.cleandoc(ast.get_docstring(node, clean=False) or ""),
                "source": src_link(mod.rel, node.lineno),
                "origin": "python",
                "bases": [x.lstrip("_") if x.startswith("_Native") else x for x in bases],
                "_node": node,
            }
        # assignment / constant
        val = None
        with contextlib.suppress(Exception):
            val = ast.unparse(node.value) if getattr(node, "value", None) is not None else None
        if val and len(val) > 160:
            val = val[:157] + "…"
        return {
            "kind": "data",
            "signature": None,
            "value": val,
            "doc": "",
            "source": src_link(mod.rel, node.lineno),
            "origin": "python",
        }

    def resolve(name: str, mod: PyModule, depth=0) -> dict | None:
        if depth > 12:
            return None
        b = mod.bindings.get(name)
        if b is None:
            return None
        if b.kind == "import":
            if b.module in ("alkahest.alkahest", "alkahest._core"):
                return native_entry(b.name, name)
            parent = load(b.module)
            pb = parent.bindings.get(b.name) if parent else None
            sub = load(f"{b.module}.{b.name}") if (pb is None or pb.kind == "import") else None
            if sub is not None and sub.dotted != mod.dotted:
                return {
                    "kind": "module",
                    "signature": None,
                    "doc": sub.doc,
                    "source": f"{GITHUB}{sub.rel}",
                    "origin": "python",
                    "_module": sub.dotted,
                }
            target = load(b.module)
            if target is None:
                return None
            return resolve(b.name, target, depth + 1)
        if b.kind == "assign" and isinstance(getattr(b.node, "value", None), ast.Name):
            alias = b.node.value.id
            if alias != name:
                r = resolve(alias, mod, depth + 1)
                if r is not None:
                    return r
        if b.kind in ("def", "class", "assign"):
            e = py_entry(b, mod)
            if e["kind"] in ("class", "exception") and not e["doc"]:
                # a Python subclass of a native class with no docstring of its own
                n = native_entry(name, name)
                if n and n["doc"]:
                    e["doc"] = n["doc"]
            return e
        return None

    def py_members(node: ast.ClassDef, mod: PyModule, qual: str) -> list[dict]:
        out = []
        for n in node.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if n.name.startswith("_") and n.name not in ("__call__",):
                    continue
                decos = _decorators(n)
                kind = "method"
                drop = True
                if "property" in decos or any(
                    d.endswith(".getter")
                    or d == "cached_property"
                    or d.endswith(".cached_property")
                    for d in decos
                ):
                    kind = "property"
                elif "staticmethod" in decos:
                    kind, drop = "staticmethod", False
                elif "classmethod" in decos:
                    kind = "classmethod"
                elif any(d.endswith(".setter") for d in decos):
                    continue
                out.append(
                    {
                        "name": n.name,
                        "qualname": f"{qual}.{n.name}",
                        "kind": kind,
                        "signature": None if kind == "property" else _py_sig(n, drop_first=drop),
                        "returns": ast.unparse(n.returns)
                        if (kind == "property" and n.returns is not None)
                        else None,
                        "doc": inspect.cleandoc(ast.get_docstring(n, clean=False) or ""),
                        "source": src_link(mod.rel, n.lineno),
                    }
                )
            elif (
                isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name)
                and not n.target.id.startswith("_")
            ):
                out.append(
                    {
                        "name": n.target.id,
                        "qualname": f"{qual}.{n.target.id}",
                        "kind": "attribute",
                        "signature": None,
                        "returns": ast.unparse(n.annotation),
                        "doc": "",
                        "source": src_link(mod.rel, n.lineno),
                    }
                )
        return out

    def finalize(qual: str, name: str, e: dict, module: str):
        members = []
        if e.get("_node") is not None and isinstance(e["_node"], ast.ClassDef):
            members = py_members(e["_node"], load(module) or root, qual)
            # a Python wrapper of a native class inherits the native members
            nat = rust_classes.get(name)
            if nat:
                have = {m["name"] for m in members}
                for m in nat.members:
                    if m.pyname not in have and not m.pyname.startswith("_"):
                        members.append(_rust_member(qual, m))
        elif e.get("_members"):
            members = [
                _rust_member(qual, m)
                for m in e["_members"]
                if not m.pyname.startswith("_") or m.pyname == "__call__"
            ]
        e.pop("_node", None)
        e.pop("_members", None)
        e.update({"name": name, "qualname": qual, "module": module, "members": members})
        symbols[qual] = e

    def _rust_member(qual: str, m: RustItem) -> dict:
        return {
            "name": m.pyname,
            "qualname": f"{qual}.{m.pyname}",
            "kind": m.kind,
            "signature": None if m.kind == "property" else m.signature,
            "returns": None,
            "doc": m.doc,
            "source": src_link(m.path, m.line),
        }

    public = list(dict.fromkeys(root.all or []))
    submodules = []
    for name in public:
        e = resolve(name, root)
        if e is None:
            e = native_entry(name, name)
        if e is None:
            e = {"kind": "data", "signature": None, "doc": "", "source": None, "origin": "unknown"}
        if e["kind"] == "module":
            submodules.append((name, e.pop("_module")))
        mod_of = "alkahest"
        finalize(name, name, e, mod_of)

    # Fallback docstrings for native exceptions: exceptions.py wrappers.
    exc_mod = load("alkahest.exceptions")
    for q, e in symbols.items():
        if (
            e["kind"] in ("class", "exception")
            and not e["doc"]
            and exc_mod
            and q in exc_mod.bindings
        ):
            b = exc_mod.bindings[q]
            if b.kind == "class":
                e["doc"] = inspect.cleandoc(ast.get_docstring(b.node, clean=False) or "")
                e["kind"] = "exception"
        if q.endswith("Error") and e["kind"] == "class":
            e["kind"] = "exception"

    # Submodule members
    for short, dotted in submodules:
        sub = load(dotted)
        if sub is None:
            continue
        names = (
            sub.all
            if sub.all is not None
            else [
                n
                for n in sub.defined_order
                if not n.startswith("_") and sub.bindings[n].kind in ("def", "class")
            ]
        )
        for n in names:
            e = resolve(n, sub)
            if e is None or e["kind"] == "module":
                continue
            finalize(f"{short}.{n}", n, e, dotted)

    # ---- pages from the Sphinx reference ----------------------------------
    pages = []
    placed: dict[str, str] = {}
    sphinx = repo / "docs" / "sphinx" / "api"
    page_specs = []
    for slug, fname, group in API_PAGES:
        if fname is None:
            spec = {"title": "Certificates API", "sections": []}
        else:
            p = sphinx / f"{fname}.rst"
            if not p.exists():
                continue
            spec = parse_rst_page(p)
        for title, names in EXTRA_SECTIONS.get(slug, []):
            present = [n for n in names.split() if n in symbols]
            if present:
                spec["sections"].append(
                    {
                        "title": title,
                        "items": [
                            {"t": "sym", "group": [("function", n)], "rst": "", "members": []}
                            for n in present
                        ],
                    }
                )
        if slug == "errors":
            known = {
                g[1].split("(")[0].removeprefix("alkahest.")
                for sec in spec["sections"]
                for it in sec["items"]
                if it["t"] == "sym"
                for g in it["group"]
            }
            exc = sorted(
                q
                for q, e in symbols.items()
                if e["kind"] == "exception" and q not in known and "." not in q
            )
            if exc:
                spec["sections"].append(
                    {
                        "title": "More exception classes",
                        "items": [
                            {"t": "sym", "group": [("exception", n)], "rst": "", "members": []}
                            for n in exc
                        ],
                    }
                )
        page_specs.append((slug, group, spec))

    def lookup(raw_name: str) -> str | None:
        n = raw_name.removeprefix("alkahest.")
        if n in symbols:
            return n
        if "." in n:
            head, tail = n.split(".", 1)
            if f"{head}.{tail}" in symbols:
                return f"{head}.{tail}"
            if tail in symbols:
                return tail
            # Class.method
            if head in symbols:
                return None
        return None

    # First pass: anchors/URLs for linking
    for slug, _group, spec in page_specs:
        for sec in spec["sections"]:
            for it in sec["items"]:
                if it["t"] != "sym":
                    continue
                for _kind, sig in it["group"]:
                    nm, _ = _split_sig(sig)
                    q = lookup(nm) or nm.removeprefix("alkahest.")
                    if q not in placed:
                        placed[q] = slug
                        LINKER.urls[q] = f"~/api/{slug}/#{q}"
    # Unplaced public symbols go to "more" pages
    more_pages: dict[str, list[str]] = {}
    for q, e in symbols.items():
        if q in placed:
            continue
        if e["kind"] == "module":
            slug = "mod-" + q.replace("_", "-")
            placed[q] = slug
            LINKER.urls[q] = f"~/api/{slug}/"
            continue
        src = e.get("source") or ""
        mod = e.get("module", "")
        if "." in q and q.split(".")[0] in dict(submodules):
            slug = "mod-" + q.split(".")[0].replace("_", "-")
        else:
            slug = next(s for s, _t, rx in _MORE_GROUPS if re.search(rx, src + " " + mod))
        more_pages.setdefault(slug, []).append(q)
        placed[q] = slug
        LINKER.urls[q] = f"~/api/{slug}/#{q}"
    for q, e in symbols.items():
        for m in e.get("members", []):
            LINKER.urls.setdefault(m["qualname"], f"~/api/{placed[q]}/#{m['qualname']}")

    # Second pass: build page JSON
    def sym_payload(q: str, rst_sig: str | None, rst_body: str, rst_members: list) -> dict:
        e = symbols.get(q)
        if e is None:
            # documented in the RST reference but not in __all__ (e.g. a method
            # or an object attribute): render from the RST alone.
            nm, sig = _split_sig(rst_sig or q)
            return {
                "qualname": q,
                "name": nm.split(".")[-1],
                "kind": "function",
                "signature": sig or None,
                "blocks": parse_docstring(rst_body),
                "summary": summary_of(rst_body),
                "source": None,
                "origin": "reference",
                "members": [
                    {
                        "qualname": f"{q}.{_split_sig(m['sig'])[0]}",
                        "name": _split_sig(m["sig"])[0],
                        "kind": m["kind"],
                        "signature": _split_sig(m["sig"])[1] or None,
                        "blocks": parse_docstring(m["rst"]),
                        "source": None,
                    }
                    for m in rst_members
                ],
            }
        doc = e.get("doc") or ""
        used_rst = False
        if not doc.strip() and rst_body.strip():
            doc, used_rst = rst_body, True
        sig = e.get("signature")
        if not sig and rst_sig:
            sig = _split_sig(rst_sig)[1] or None
        members = []
        rst_by_name = {}
        for m in rst_members:
            nm, msig = _split_sig(m["sig"])
            rst_by_name[nm] = (m, msig)
        seen = set()
        for m in e.get("members", []):
            mdoc = m.get("doc") or ""
            rm = rst_by_name.get(m["name"])
            if not mdoc.strip() and rm:
                mdoc = rm[0]["rst"]
            msig = m.get("signature")
            if (not msig or msig == "()") and rm and rm[1]:
                msig = rm[1]
            members.append(
                {
                    "qualname": m["qualname"],
                    "name": m["name"],
                    "kind": m["kind"],
                    "signature": msig,
                    "returns": m.get("returns"),
                    "blocks": parse_docstring(mdoc),
                    "summary": summary_of(mdoc),
                    "source": m.get("source"),
                }
            )
            seen.add(m["name"])
        for nm, (m, msig) in rst_by_name.items():
            # RST-only members are kept only when the code gave us nothing:
            # the hand-written pages list methods that no longer exist.
            if nm in seen or e.get("members"):
                continue
            members.append(
                {
                    "qualname": f"{q}.{nm}",
                    "name": nm,
                    "kind": m["kind"],
                    "signature": msig or None,
                    "returns": None,
                    "blocks": parse_docstring(m["rst"]),
                    "summary": summary_of(m["rst"]),
                    "source": None,
                }
            )
        # members with docs first is wrong — keep source order, but drop dunders
        members = [
            m for m in members if not (m["name"].startswith("__") and m["name"] != "__call__")
        ]
        blocks = parse_docstring(doc)
        return {
            "qualname": q,
            "name": e["name"],
            "kind": e["kind"],
            "signature": sig,
            "value": e.get("value"),
            "bases": e.get("bases"),
            "blocks": blocks,
            "summary": summary_of(doc),
            "source": e.get("source"),
            "origin": e.get("origin"),
            "from_reference": used_rst,
            "members": members,
        }

    emitted: set[str] = set()
    for slug, group, spec in page_specs:
        psecs = []
        for sec in spec["sections"]:
            items = []
            for it in sec["items"]:
                if it["t"] == "prose":
                    md = rst_to_md(it["rst"])
                    if md:
                        items.append({"t": "prose", "md": md})
                    continue
                group_syms = []
                for _kind, sig in it["group"]:
                    nm, _ = _split_sig(sig)
                    q = lookup(nm) or nm.removeprefix("alkahest.")
                    if q in emitted:
                        # documented twice in the RST; link to the first
                        group_syms.append({"ref": q})
                        continue
                    emitted.add(q)
                    group_syms.append(
                        sym_payload(
                            q, sig, it["rst"] if len(it["group"]) == 1 else "", it["members"]
                        )
                    )
                # grouped signatures share one RST body: give it to members
                # that have nothing of their own
                if len(it["group"]) > 1 and it["rst"].strip():
                    shared = parse_docstring(it["rst"])
                    lacking = [g for g in group_syms if "ref" not in g and not g["blocks"]]
                    if lacking:
                        lacking[-1]["blocks"] = shared
                        lacking[-1]["from_reference"] = True
                        if not lacking[-1]["summary"]:
                            lacking[-1]["summary"] = summary_of(it["rst"])
                        for g in lacking[:-1]:
                            g["see_below"] = lacking[-1]["qualname"]
                items.append({"t": "syms", "syms": group_syms})
            psecs.append({"title": sec["title"], "items": items})
        pages.append({"slug": slug, "group": group, "title": spec["title"], "sections": psecs})

    titles = {s: t for s, t, _ in _MORE_GROUPS}
    sub_docs = {short: load(dotted).doc for short, dotted in submodules if load(dotted)}
    for slug in sorted(more_pages, key=lambda s: (s.startswith("mod-"), s == "more", s)):
        qs = more_pages[slug]
        if slug.startswith("mod-"):
            short = qs[0].split(".")[0]
            title = f"alkahest.{short}"
            intro = rst_to_md(sub_docs.get(short, "") or "")
            group = "Submodules"
        else:
            title = titles[slug]
            intro = ""
            group = "More"
        # sort: classes, then functions, then data; alphabetical inside
        order = {"class": 0, "exception": 1, "function": 2, "module": 3, "data": 4}
        qs = sorted(qs, key=lambda q: (order.get(symbols[q]["kind"], 5), q.lower()))
        items = []
        if intro:
            items.append({"t": "prose", "md": intro})
        for q in qs:
            items.append({"t": "syms", "syms": [sym_payload(q, None, "", [])]})
            emitted.add(q)
        pages.append(
            {
                "slug": slug,
                "group": group,
                "title": title,
                "sections": [{"title": None, "items": items}],
            }
        )

    stats = {
        "public": len(public),
        "symbols": len(symbols),
        "with_doc": sum(1 for e in symbols.values() if (e.get("doc") or "").strip()),
        "native_functions": len(rust_fns),
        "native_classes": len(rust_classes),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"pages": pages, "stats": stats}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"api: {stats}", file=sys.stderr)


if __name__ == "__main__":
    main()
