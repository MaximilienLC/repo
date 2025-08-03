from dataclasses import dataclass
from typing import Annotated as An

from hydra import conf as hc
from hydra import types as ht
from hydra.experimental.callbacks import LogJobReturnCallback
from hydra_zen import make_config

from common.utils.beartype import ge, not_empty, one_of
from common.utils.hydra_zen import generate_config


@dataclass
class BaseSublogicConfig:

    output_dir: An[str, not_empty()] = "${hydra:runtime.output_dir}"
    data_dir: An[str, not_empty()] = "${oc.env:REPO_PATH}/data/"
    device: An[str, one_of("cpu", "gpu")] = "cpu"
    seed: An[int, ge(0)] = 0


@dataclass
class BaseHydraConfig(
    make_config(  # type: ignore[misc]
        bases=(hc.HydraConf,),
        callbacks={"log_job_return": generate_config(LogJobReturnCallback)},
        job=hc.JobConf(
            config=hc.JobConf.JobConfig(
                override_dirname=hc.JobConf.JobConfig.OverrideDirname(
                    kv_sep="~",
                    item_sep="#",
                    exclude_keys=[
                        "logic",
                        "project",
                        "trainer.max_epochs",
                        "trainer.max_steps",
                    ],
                ),
            ),
        ),
        mode=ht.RunMode.MULTIRUN,
        sweep=hc.SweepDir(
            dir="${oc.env:REPO_PATH}/data/${project}/${logic}/",
            subdir="overrides#${hydra:job.override_dirname}/",
        ),
    ),
): ...