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
  - [Codebase conventions](#codebase-conventions)
    - [Type hinting](#type-hinting)
      - [Enforcement with `beartype`](#enforcement-with-beartype)
      - [`torch` tensors](#torch-tensors)
      - [`beartype` validators](#beartype-validators)
      - [Type hinting variables](#type-hinting-variables)
    - [`torch` tensor manipulation](#torch-tensor-manipulation)
- [Miscelleaneous](#miscelleaneous)
  - [Computing environment](#computing-environment)
- [Gotchas](#gotchas)

# Instructions for Claude Code

Make absolutely sure that nothing read from `.md` files is ever ignored, even when debugging.

## Context management

Your effective context size is ~140k tokens. Be mindful to allocate it properly.

### File reading

Never read files that you were not refered to (`@file.md` <=> refered, `file.md` <=> not refered).
The only exception are files in the `/utils/` folder.

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

## Codebase conventions

### Type hinting

Type hints are used extensively in this codebase.

#### Enforcement with `beartype`

Type hints are enforced using the python package `beartype`.

The following code snippet can be found in relevant `__init__.py` files.
```
from beartype.claw import beartype_this_package
beartype_this_package()
```

#### `torch` tensors

In order to enhance correctness, we further constrain `torch` tensors using the python package `jaxtyping`.

```
from jaxtyping import Float, Int

def compute_loss(
  predicted_logits: Float[Tensor, " BS SL NL"],
  target_features: Float[Tensor, " BS SL NTF"],
) -> Float[Tensor, " "]: ...
```

#### `beartype` validators

`@/utils/beartype.py` provides several `BeartypeValidator` generating functions.

They are used as such:
```
from typing import Annotated as An
from utils.beartype import not_empty, equal, one_of, ge, gt, le, lt

class Test:
    input_size: An[int, ge(1)]
```

#### Type hinting variables

In addition to type hinting arguments, return values, etc. as is common practice; we also type hint variables

Example:

```
flat_indices: Int[Tensor, " BSxSL 1"] = torch.multinomial(
    input=flat_pi,
    num_samples=1,
)
```

### `torch` tensor manipulation

We use python package `einops` in order to manipulate tensors, e.g.

```
from einops import rearrange, reduce, repeat
flat_pi: Float[Tensor, " BSxSL NG"] = rearrange(
    tensor=pi,
    pattern="BS SL NG -> (BS SL) NG",
)
```

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