# code/core/

## Overview

This folder contains core code, as in code that is likely to be reused across experiments.

It is further subdivided into the two machine learning paradigms used in this library:
- Deep Learning over at @code/core/deeplearning/
- Neuroevolution in @code/core/neuroevolution/

## Implementation Details

All numeral values (rank 1+ tensors) are processesd in PyTorch, with all operations being performed on the select GPUs.