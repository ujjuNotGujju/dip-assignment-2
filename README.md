# DIP Assignment 2

Digital Image Processing, Assignment 2. Full spec: [docs/DIP_A2.pdf](docs/DIP_A2.pdf).
Released 2026-10-08, due 2026-10-29. Report+code 50%, viva 50%; **viva mark caps the
report mark per problem**.

## Progress

| # | Problem | Status |
|---|---|---|
| 1 | DFT and the machinery of frequency | **done** |
| 2 | Frequency-domain filtering | scaffolded |
| 3 | Homomorphic filtering | scaffolded |
| 4 | Edge-preserving smoothing | scaffolded |
| 5 | Denoising and the Wiener filter | scaffolded |
| 6 | Motion blur and deblurring | scaffolded |
| 7 | Lens distortion | blocked — spec missing, see `problems/problem7_lens_distortion/README.md` |

## Rules (strict — see the PDF for full wording)

- **From scratch = NumPy array arithmetic.** `scipy.ndimage`, `scipy.signal`, `cv2`,
  `skimage` must **not** be used for any operation a problem asks us to implement.
  They may only be used to validate/compare against our own implementation, and only
  where a subtask explicitly asks for that comparison.
- `numpy.fft` is allowed everywhere (we implement the DFT matrix ourselves once in
  Problem 1; elsewhere `numpy.fft` is fine to use directly).
- Every number in the report must come from the machine that ran the code — CPU, RAM,
  and NumPy version are stated per problem.
- Where the PDF asks for a dated, pre-implementation written prediction, that file is
  written and committed **before** the corresponding experiment is run.
- Problem-specific subtasks call for self-synthesized images (not external ones) or
  self-captured photos — see each problem's README.

> **Note:** the PDF's table of contents lists **Problem 7 — Lens Distortion (10 pts)**
> but the document contains no detailed problem text for it. Flagged in
> `problems/problem7_lens_distortion/README.md` — confirm with the instructor before
> starting that problem.

## Structure

```
docs/            assignment PDF
data/
  provided/      images supplied with the assignment, one subfolder per problem
  synthetic/     images we generate ourselves (sinusoids, step edges, etc.)
  own_captures/  self-captured photos required by Problems 5 and 6
src/dip_a2/      shared Python package (DFT machinery, filters, metrics, ...)
problems/        one folder per problem, with its own README/checklist, code,
                 dated predictions (where required), and results
report/          the written report (report.md), figures referenced from it
```

## Environment

Captured for real by `problems/problem1_dft/problem1.py` (see `report/report.md` and
`problems/problem1_dft/results/machine_info.json`):

- CPU: Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz (8 logical CPUs)
- RAM: 14.96 GB
- NumPy version: 2.5.4

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
