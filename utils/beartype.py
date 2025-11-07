from beartype.vale import Is
from beartype.vale._core._valecore import BeartypeValidator


def not_empty() -> BeartypeValidator:

    def _not_empty(x: object) -> bool:
        return (isinstance(x, str) or isinstance(x, list)) and len(x) > 0

    return Is[lambda x: _not_empty(x)]


def equal(element: object) -> BeartypeValidator:

    def _equal(x: object, element: object) -> bool:
        return x == element

    return Is[lambda x: _equal(x, element)]


def one_of(*elements: object) -> BeartypeValidator:

    def _one_of(x: object, elements: tuple[object, ...]) -> bool:
        return x in elements

    return Is[lambda x: _one_of(x, elements)]


def ge(val: float) -> BeartypeValidator:

    def _ge(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x >= val

    return Is[lambda x: _ge(x, val)]


def gt(val: float) -> BeartypeValidator:

    def _gt(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x > val

    return Is[lambda x: _gt(x, val)]


def le(val: float) -> BeartypeValidator:

    def _le(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x <= val

    return Is[lambda x: _le(x, val)]


def lt(val: float) -> BeartypeValidator:

    def _lt(x: object, val: float) -> bool:
        return isinstance(x, int | float) and x < val

    return Is[lambda x: _lt(x, val)]