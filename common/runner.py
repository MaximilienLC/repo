from abc import ABC, abstractmethod
from typing import Any, final

from hydra_zen import ZenStore, zen
from omegaconf import OmegaConf

from common.utils.comet import login_comet

from .config import BaseHydraConfig
from .utils.hydra_zen import destructure
from .utils.runner import (
    get_absolute_project_path,
    get_logic_name,
    get_project_name,
)


class BaseLogicRunner(ABC):

    hydra_config = BaseHydraConfig

    @final
    @classmethod
    def handle_configs(cls: type["BaseLogicRunner"]) -> None:
        store = ZenStore()
        store(cls.hydra_config, name="config", group="hydra")
        store({"project": get_project_name()}, name="project")
        store({"logic": get_logic_name()}, name="logic")
        # Hydra runtime type checking issues with structured configs:
        # https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/621#discussioncomment-7938326
        # `destructure` disables Hydra's runtime type checking, which is
        # fine since we use Beartype throughout the codebase.
        store = store(to_config=destructure)
        cls.store_configs(store=store)
        store.add_to_hydra_store(overwrite_ok=True)

    @final
    @classmethod
    def run_logic(cls: type["BaseLogicRunner"]) -> None:
        OmegaConf.register_new_resolver("eval", eval)
        login_comet()
        cls.handle_configs()
        zen(cls.run_sublogic).hydra_main(
            config_path=get_absolute_project_path(),
            config_name="config",
            version_base=None,
        )

    @classmethod
    @abstractmethod
    def store_configs(
        cls: type["BaseLogicRunner"],
        store: ZenStore,
    ) -> None: ...

    @staticmethod
    @abstractmethod
    def run_sublogic(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        ...
