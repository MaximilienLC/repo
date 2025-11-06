- [Instructions for Claude Code](#instructions-for-claude-code)
  - [Context management](#context-management)
    - [File reading](#file-reading)
  - [Task execution](#task-execution)
  - [`CLAUDE.md` file editing](#claudemd-file-editing)
- [Specifications](#specifications)
  - [Codebase structure](#codebase-structure)
    - [Focus on `CLAUDE.md` files](#focus-on-claudemd-files)
    - [`CLAUDE.md` file structure](#claudemd-file-structure)
      - [Table of contents](#table-of-contents)
      - [# Specifications](#-specifications)
- [Miscelleaneous](#miscelleaneous)
  - [Computing environment](#computing-environment)
- [Gotchas](#gotchas)

# Instructions for Claude Code

Make absolutely sure that nothing read from `.md` files is ever ignored, even when debugging.

## Context management

Your effective context size is ~140k tokens. Be mindful to allocate it properly.

### File reading

Never read files that you were not refered to (`@file.md` <=> refered, `file.md` <=> not refered)

## Task execution

Never run any model optimization as a background task.

## `CLAUDE.md` file editing

You may edit any part of `CLAUDE.md` files as you see fit.

# Specifications

## Codebase structure

### Focus on `CLAUDE.md` files

Source code files are second-class citizens in this codebase.
`CLAUDE.md` files contain all of the necessary information to (re)generate the source code files that drive execution.

### `CLAUDE.md` file structure

#### Table of contents

The top of `CLAUDE.md` files always feature a table of contents of format:

```
- [Header 1](#header-1)
  - [Subheader 1](#subheader-1)
    - [Subsubheader 1](#subsubheader-1)
  - [Subheader 2](#subheader-2)
```

#### # Specifications

The **Specifications** sections contain all of the relevant information in order for Claude Code to generate the source code files in that folder (though not its subfolders). 

# Miscelleaneous

## Computing environment

- Windows 11
- GPU: AMD Radeon RX 7800 XT (16GB VRAM)
- CPU: AMD Ryzen 7 7700X (8 core, 16 threads)
- RAM: 32GB, 6000 MT/s
- SSD: NVME

`torch`, `torchvision` and `torchaudio` are installed locally with GPU-support.
Make sure that all tensor operations are ran on the GPU.
Do not investigate `amdsmi` warnings.

# Gotchas

N/A