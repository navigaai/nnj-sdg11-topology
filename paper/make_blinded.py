"""Generate the BLINDED Springer single-file manuscript for double-blind submission.

Source files (main-springer.tex, abstract.tex, body.tex) keep the real author details
for the camera-ready; this script inlines them and strips every identifying element,
writing paper/sn-article-blinded.tex. Author identity lives only in the separate
title-page.tex. Figures become \fbox placeholders (Overleaf file-upload workaround).

Usage: uv run python paper/make_blinded.py   (run from repo root or paper/)
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _fig_placeholder(m: re.Match) -> str:
    name = m.group(1).replace("_", r"\_")
    return (r"\fbox{\parbox[c][4.5cm][c]{0.72\linewidth}{\centering\ttfamily " + name
            + r"\\[2pt]\normalfont\itshape (drag-drop this PDF into figures/, then "
            + r"restore \textbackslash includegraphics)}}")


def main() -> None:
    main_tex = (HERE / "main-springer.tex").read_text()
    abstract = (HERE / "abstract.tex").read_text().strip()
    body = (HERE / "body.tex").read_text().strip()

    out = (main_tex
           .replace(r"\abstract{\input{abstract}}", r"\abstract{%s}" % abstract)
           .replace(r"\input{body}", body)
           .replace(r"\bibliography{references}", r"\bibliography{sn-bibliography}"))

    # --- blinding transforms ---
    # author + email
    out = out.replace(
        r"\author[1]{\fnm{Seyda} \sur{Emekci}}\email{semekci@aybu.edu.tr}",
        r"\author[1]{\fnm{}\sur{[Author name omitted for double-blind peer review]}}")
    # affiliation block (exact three source lines; nested braces defeat a simple regex)
    affil_src = ("\\affil[1]{\\orgdiv{Department of Architecture},\n"
                 "\\orgname{Ankara Y{\\i}ld{\\i}r{\\i}m Beyaz{\\i}t University},\n"
                 "\\city{Ankara}, \\country{Turkey}}")
    assert affil_src in out, "affiliation block not found verbatim"
    out = out.replace(affil_src,
                      r"\affil[1]{[Affiliation omitted for double-blind peer review]}")
    # repository URLs (identify the author's org)
    out = re.sub(r"\\url\{https://github\.com/navigaai/nnj-sdg11-topology\}",
                 r"[repository URL omitted for double-blind peer review]", out)
    # author-contributions name + ORCID
    out = out.replace(r"S.\ Emekci (ORCID 0000-0002-5470-6485) conceived",
                      r"The author conceived")
    # drop comment-only lines (they can carry identifying notes, e.g. the ORCID hint)
    out = "\n".join(ln for ln in out.splitlines()
                    if not re.match(r"^\s*%", ln))

    # sanity: no identifying strings remain
    for token in ["Emekci", "aybu", "navigaai", "0000-0002-5470-6485",
                  "Ankara", "Department of Architecture"]:
        assert token not in out, f"identifying token still present: {token}"

    real = out  # (real-figure version, if ever needed)
    ph = re.sub(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\.pdf\}",
                _fig_placeholder, out)

    (HERE / "sn-article-blinded.tex").write_text(ph)
    (Path("/tmp/sn-article-blinded.tex")).write_text(ph)
    print(f"wrote paper/sn-article-blinded.tex ({len(ph)} chars, "
          f"{real.count('includegraphics')} figures as placeholders); blind-safe.")


if __name__ == "__main__":
    main()
