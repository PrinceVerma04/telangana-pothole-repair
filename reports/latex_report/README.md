# RoadFix project report (LaTeX)

- `main.tex`: the report source (all diagrams are TikZ, so no image files are needed).
- `main.pdf`: the compiled report (14 pages), built with MiKTeX `pdflatex` on 14 November 2026.

## Compile
```bash
pdflatex main.tex
pdflatex main.tex      # second run fills in the table of contents and cross-references
```
or `latexmk -pdf main.tex`. It needs a standard TeX distribution (MiKTeX or TeX Live) with these packages:
`lmodern`, `microtype`, `geometry`, `amsmath`, `booktabs`, `tabularx`, `longtable`, `xcolor`, `enumitem`,
`fancyhdr`, `caption`, `tikz` (pgf), `hyperref`. Overleaf also compiles it as-is (upload `main.tex`).

## Contents
1. Status at a glance
2. The problem
3. System architecture
4. How the system works, step by step (13 steps, from camera calibration to bus-verified repair and DLP)
5. The simulator and the end-to-end demo (results of the 14 Nov 2026 run)
6. Evidence: what has been measured
7. Status by work package (WP0-WP12)
8. Pending steps, with effort estimates and a timeline
9. How to run it
- Appendices: key documents, glossary

Numbers come from `reports/REPORT.md`, `reports/demo_summary.json` and `docs/DECISIONS.md`. Durations for future
work are estimates.
