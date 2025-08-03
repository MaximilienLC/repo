import sys
from pathlib import Path


def get_logic_name() -> str:
    has_logic_arg = False
    for arg in sys.argv:
        if arg.startswith("logic="):
            has_logic_arg = True
            logic_name = arg.split("=", maxsplit=1)[-1]
    if not has_logic_arg:
        error_msg = (
            "You must specify the ``logic`` argument in the form"
            "``logic=my_logic``."
        )
        raise RuntimeError(error_msg)
    return logic_name


def get_project_name() -> str:
    main_file = str(sys.modules["__main__"].__file__)
    main_file_path = Path(main_file).resolve()
    return str(main_file_path.parent.name)


def get_absolute_project_path() -> str:
    main_file = str(sys.modules["__main__"].__file__)
    main_file_path = Path(main_file).resolve()
    main_file_parent_path = main_file_path.parent
    if not (main_file_parent_path / "logic").exists():
        error_msg = (
            "The main file must be located in the same directory as the "
            "``logic`` directory."
        )
        raise RuntimeError(error_msg)
    return str(main_file_parent_path)
