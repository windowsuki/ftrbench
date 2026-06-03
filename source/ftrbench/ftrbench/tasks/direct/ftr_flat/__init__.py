# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .ftr_flat_env import (
    FtrFlatEnv,
    FtrFlatEnvCfg,
    FtrReferenceRoughEnvCfg,
    FtrReferenceTerrainEnvCfg,
    FtrRoughEnvCfg,
)


gym.register(
    id="Ftr-Flat-Direct-v0",
    entry_point="ftrbench.tasks.direct.ftr_flat.ftr_flat_env:FtrFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FtrFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FtrFlatPPORunnerCfg",
    },
)

gym.register(
    id="Ftr-Rough-Direct-v0",
    entry_point="ftrbench.tasks.direct.ftr_flat.ftr_flat_env:FtrFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FtrRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FtrRoughPPORunnerCfg",
    },
)

gym.register(
    id="Ftr-ReferenceRough-Direct-v0",
    entry_point="ftrbench.tasks.direct.ftr_flat.ftr_flat_env:FtrFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FtrReferenceRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FtrReferenceRoughPPORunnerCfg",
    },
)

gym.register(
    id="Ftr-ReferenceTerrain-Direct-v0",
    entry_point="ftrbench.tasks.direct.ftr_flat.ftr_flat_env:FtrFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FtrReferenceTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FtrReferenceTerrainPPORunnerCfg",
    },
)

__all__ = [
    "FtrFlatEnv",
    "FtrFlatEnvCfg",
    "FtrRoughEnvCfg",
    "FtrReferenceRoughEnvCfg",
    "FtrReferenceTerrainEnvCfg",
]
