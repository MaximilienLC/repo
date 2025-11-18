# Codebase conventions
- [Codebase conventions](#codebase-conventions)
  - [Type hinting](#type-hinting)
    - [Type hints enforcement using `beartype`](#type-hints-enforcement-using-beartype)
    - [`torch` tensors with `jaxtyping`](#torch-tensors-with-jaxtyping)
    - [`beartype` validators](#beartype-validators)
    - [Type hinting variables](#type-hinting-variables)
  - [`einops` to manipulate `torch` tensors](#einops-to-manipulate-torch-tensors)

## Type hinting

Type hints are used extensively in this codebase.

### Type hints enforcement using `beartype`

The following code snippet can be found in relevant `__init__.py` files.

```
from beartype import BeartypeConf
from beartype.claw import beartype_this_package

beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))
```

### `torch` tensors with `jaxtyping`

```
from jaxtyping import Float, Int

def compute_loss(
  predicted_logits: Float[Tensor, "BS SL NL"],
  target_features: Float[Tensor, "BS SL NTF"],
) -> Float[Tensor, "1"]: ...
```

### `beartype` validators

`utils/beartype.py` provides several `BeartypeValidator` generating functions.

```
from typing import Annotated as An
from utils.beartype import not_empty, equal, one_of, ge, gt, le, lt

class Test:
    input_size: An[int, ge(1)]
```

### Type hinting variables

In addition to type hinting arguments, return values, etc. as is common practice; we also most often type hint variables

```
flat_indices: Int[Tensor, " BSxSL 1"] = torch.multinomial(
    input=flat_pi,
    num_samples=1,
)
node_list: list[Node] = []
```

## `einops` to manipulate `torch` tensors

```
from einops import rearrange, reduce, repeat
flat_pi: Float[Tensor, " BSxSL NG"] = rearrange(
    tensor=pi,
    pattern="BS SL NG -> (BS SL) NG",
)
```