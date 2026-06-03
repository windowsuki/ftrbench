# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loader for the reference FTR terrain assets migrated to Isaac Lab 4.5."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
import yaml


class ReferenceTerrain:
    """Loads a reference terrain USD plus its start/target reset metadata."""

    def __init__(self, name: str, prim_path: str = "/World/ground"):
        self.name = name
        self.prim_path = prim_path
        self.asset_dir = Path(__file__).resolve().parent

        config_path = self.asset_dir / "config" / f"{name}.yaml"
        if not config_path.exists():
            available = ", ".join(self.list_all())
            raise FileNotFoundError(f"Reference terrain config not found: {config_path}. Available: {available}")

        with config_path.open("r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)

        self.obstacles = self._load_obstacles()
        self.births = self._load_births()

    @classmethod
    def list_all(cls) -> list[str]:
        terrain_dir = Path(__file__).resolve().parent
        config_dir = terrain_dir / "config"
        return sorted(path.stem for path in config_dir.glob("*.yaml"))

    def spawn(self, stage) -> None:
        """Spawn the terrain USD references and apply the original config attributes."""

        for name, obstacle in self.obstacles.items():
            prim_path = f"{self.prim_path}/{name}"
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=str(obstacle["path"]),
                scale=self._as_tuple(obstacle.get("scale")) if "scale" in obstacle else None,
            )
            spawn_cfg.func(
                prim_path,
                spawn_cfg,
                translation=self._as_tuple(obstacle.get("position", (0.0, 0.0, 0.0))),
                orientation=self._orientation_to_quat(obstacle.get("orient", (1.0, 0.0, 0.0, 0.0))),
            )

        for attr_cfg in self.config.get("prim_config", {}).get("set_attrs", []):
            prim_path = f"{self.prim_path}/{attr_cfg['prim_path']}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Reference terrain prim not found: {prim_path}")

            attr = prim.GetAttribute(attr_cfg["attr_name"])
            if not attr.IsValid():
                raise RuntimeError(f"Reference terrain attribute not found: {prim_path}.{attr_cfg['attr_name']}")

            attr.Set(self._usd_attr_value(attr_cfg["attr_name"], attr_cfg["value"]))

    def _load_obstacles(self) -> dict[str, dict[str, Any]]:
        obstacles = {}
        if "obstacles" in self.config:
            for name, obstacle in self.config["obstacles"].items():
                item = obstacle.copy()
                item["path"] = self.asset_dir / item["path"]
                if not item["path"].exists():
                    raise FileNotFoundError(f"Reference terrain USD not found: {item['path']}")
                obstacles[name] = item
        else:
            usd_path = self.asset_dir / "usd" / f"{self.name}.usd"
            if not usd_path.exists():
                raise FileNotFoundError(f"Reference terrain USD not found: {usd_path}")
            obstacles["terrain"] = {"path": usd_path}
        return obstacles

    def _load_births(self) -> list[dict[str, Any]]:
        if "task_info" in self.config:
            return [self.config["task_info"]]

        birth_path = self.asset_dir / "birth" / f"{self.name}.json"
        if not birth_path.exists():
            raise FileNotFoundError(f"Reference terrain birth file not found: {birth_path}")
        with birth_path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def _as_tuple(value: Any) -> tuple[float, ...]:
        return tuple(float(item) for item in value)

    @classmethod
    def _orientation_to_quat(cls, value: Any) -> tuple[float, float, float, float]:
        values = cls._as_tuple(value)
        if len(values) == 4:
            return values
        if len(values) != 3:
            raise ValueError(f"Orientation must have 3 Euler values or 4 quaternion values. Got: {values}")

        roll, pitch, yaw = (math.radians(angle) for angle in values)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        return (
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        )

    @staticmethod
    def _usd_attr_value(attr_name: str, value: Any) -> Any:
        if attr_name == "xformOp:orient":
            from pxr import Gf

            return Gf.Quatd(*value)
        if isinstance(value, list):
            return tuple(value)
        return value
