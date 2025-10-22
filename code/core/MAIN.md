# code/core/

## Overview

This folder contains core code, as in code that is likely to be reused across projects.

It is presently further subdivided into the two machine learning paradigms currently used in this library:
- `@code/core/deeplearning/`
- `@code/core/neuroevolution/`

## Implementation Details

All numeral values (rank 1+ tensors) are processed in PyTorch, with all operations being performed on the select GPUs.