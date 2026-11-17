"""Hapus komentar & ringkas docstring tanpa mengubah kode."""
import io
import re
import sys
import tokenize


def clean(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines(keepends=True)
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))

    def pos(row, col):
        return offsets[row - 1] + col

    edits = []
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    prev_sig = None
    for t in toks:
        if t.type == tokenize.COMMENT:
            start = pos(*t.start)
            end = pos(*t.end)
            line_start = offsets[t.start[0] - 1]
            if src[line_start:line_start + t.start[1]].strip() == "":
                # komentar mengisi baris penuh -> hapus barisnya
                line_end = offsets[t.start[0]]
                edits.append((line_start, line_end, ""))
            else:
                # komentar di akhir kode -> hapus komentar + spasi sebelumnya
                s = start
                while s > line_start and src[s - 1] in " \t":
                    s -= 1
                edits.append((s, end, ""))
            continue
        if t.type == tokenize.STRING and prev_sig in (
                tokenize.NEWLINE, tokenize.INDENT, tokenize.NL, None):
            body = t.string
            if body.startswith(('"""', "'''")):
                inner = body[3:-3].strip()
                first = inner.splitlines()[0].strip() if inner else ""
                q = '"""' if body.startswith('"""') else "'''"
                rep = f"{q}{first}{q}" if first else ""
                edits.append((pos(*t.start), pos(*t.end), rep))
        if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                          tokenize.DEDENT, tokenize.COMMENT, tokenize.ENCODING):
            prev_sig = t.type

    for s, e, rep in sorted(edits, key=lambda x: x[0], reverse=True):
        src = src[:s] + rep + src[e:]

    src = re.sub(r"[ \t]+\n", "\n", src)
    src = re.sub(r"\n{3,}", "\n\n", src)
    src = re.sub(r"\n{2,}$", "\n", src)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("cleaned", path)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        clean(p)
