# Building the Nexus Network Journal (Springer) version

The manuscript exists in two builds that **share the same content** (`abstract.tex`,
`body.tex`, `references.bib`, `figures/`):

| File | Class | Purpose |
|------|-------|---------|
| `main.tex` | `article` | Portable; compiles anywhere with pdflatex. Used for drafting/review. |
| `main-springer.tex` | `sn-jnl` | **Submission build** for NNJ. Needs the Springer Nature class. |

Editing `abstract.tex` / `body.tex` / `references.bib` updates **both** builds — no
content is duplicated.

## Why the Springer build is not compiled in this repo

The Springer Nature unified class `sn-jnl.cls` (+ its `.bst`) is **not bundled**
here and could not be fetched in the build environment (CTAN mirrors returned HTML;
`tlmgr` mirror unreachable). It is trivially available two ways:

### Option A — Overleaf (recommended, zero setup)
1. Overleaf → **New Project → Templates** → search **"Springer Nature LaTeX Template"**
   (the official `sn-jnl` template; the Nexus Network Journal uses it).
2. In that project, replace the template's `sn-article.tex` with **`main-springer.tex`**
   from this folder, and upload `abstract.tex`, `body.tex`, `references.bib`, and the
   `figures/` directory.
3. Recompile. The class file is already present in the Overleaf template.

### Option B — Local install
```bash
tlmgr install sn-jnl            # when a working TeX mirror is available
# ...or unzip sn-jnl.cls + sn-jnl.bst from the Springer template into this folder
cd paper && latexmk -pdf main-springer.tex
```

## Front-matter mapping (already done in main-springer.tex)

| Content | article (`main.tex`) | Springer (`main-springer.tex`) |
|---------|----------------------|-------------------------------|
| Title | `\title{...}` | `\title[short]{...}` |
| Author | `\author{... \\ affil}` | `\author[1]{\fnm{}\sur{}}` + `\affil[1]{\orgdiv{}\orgname{}...}` |
| Abstract | `\begin{abstract}\input{abstract}` | `\abstract{\input{abstract}}` |
| Keywords | — | `\keywords{...}` |
| Bib style | `\bibliographystyle{plainnat}` | class-provided (`sn-mathphys-num`) |

## Reference style

`main-springer.tex` uses `sn-mathphys-num` (numbered Vancouver style), which fits
NNJ's mathematics-facing content. If the handling editor requests author–year,
change the class option to `sn-basic`. Verify the final choice against the current
NNJ author instructions before submission.

## Pre-submission checklist (NNJ-specific)

- [ ] Confirm the current NNJ reference style (`sn-mathphys-num` vs `sn-basic`).
- [ ] Author name / affiliation / email correct (`main-springer.tex`).
- [ ] All 12 references still resolve (they are CrossRef/arXiv-verified; re-check
      the 2 official/software cites: UN SDG resolution, GUDHI).
- [ ] Figures are vector where possible (Fig. 5/6 are currently 200-dpi PNG; the
      pipeline can re-emit PDF via `matplotlib` `savefig(..., format="pdf")`).
- [ ] Add ORCID, funding, data-availability, and conflict-of-interest statements
      as required by Springer.
