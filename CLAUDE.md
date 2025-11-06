# /

- [/](#)
  - [Computing environment](#computing-environment)
  - [Instructions](#instructions)
  - [Structure](#structure)

## Computing environment

- Windows 11
- GPU: AMD Radeon RX 7800 XT (16GB VRAM)
- CPU: AMD Ryzen 7 7700X (8 core, 16 threads)
- RAM: 32GB, 6000 MT/s
- SSD: NVME

`torch`, `torchvision` and `torchaudio` are installed locally with GPU-support.
Make sure that all tensor operations are ran on the GPU.
Do not investigate `amdsmi` warnings.

## Instructions

Never read any file that you were not refered to.

Absolutely make sure that no specification/prompt is ignored.

All documentation is written in markdown.

All folders can possess a `docs/` subfolder.

Whenever asked to work on a file in a given folder, make sure to read all the markdown files present in its `docs/` subfolder.

There will be times where the code you write will not run or work properly. Whenever you find a fix for it, make sure you create/update `docs/GOTCHAS.md` pertaining to the file you implemented the fix in.

Do not run any tasks in the background.

## Structure

This directory contains the repository's codebase.

It is divided into two subdirectories:
- `@projects/`, that contains project-specific directories.
- `@neuroevolution/`, that contains code that is likely to be reused across projects.