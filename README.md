# Assessing the potential of few-shot prompt engineering for text-to-code

Code associated with research paper.

To generate the figures in the paper:

- `make code` to generate code with transformer model
- `make evaluation` to run test suite on generated code (recommend doing this in some form of sandbox - it will need write access to `results/evaluation` to save its output)
- `make figures` to generate figures (in `results/figures`).