from beartype.vale import Is
from beartype.vale._core._valecore import BeartypeValidator


def not_empty() -> BeartypeValidator:

    def _not_empty(x: object) -> bool:
        return isinstance(x, str) and len(x) > 0

    return Is[lambda x: _not_empty(x)]


def equal(element: object) -> BeartypeValidator:

    def _equal(x: object, element: object) -> bool:
        return x == element

    return Is[lambda x: _equal(x, element)]


def one_of(*elements: object) -> BeartypeValidator:

    def _one_of(x: object, elements: tuple[object, ...]) -> bool:
        return x in elements

    return Is[lambda x: _one_of(x, elements)]


def greater_or_equal(val: float) -> BeartypeValidator:

    def _greater_or_equal(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x >= val

    return Is[lambda x: _greater_or_equal(x, val)]


def greater_than(val: float) -> BeartypeValidator:

    def _greater_than(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x > val

    return Is[lambda x: _greater_than(x, val)]


def less_or_equal(val: float) -> BeartypeValidator:

    def _less_or_equal(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x <= val

    return Is[lambda x: _less_or_equal(x, val)]


def less_than(val: float) -> BeartypeValidator:

    def _less_than(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x < val

    return Is[lambda x: _less_than(x, val)]
