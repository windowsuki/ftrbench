# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Current Milestone

The repository now includes a first-pass FTR import target for Isaac Sim 4.5:

- FTR robot USD copied into `source/ftrbench/ftrbench/assets/usd/ftr/ftr_v1.usd`
- A minimal direct environment registered as `Ftr-Flat-Direct-v0`
- Basic track/flipper control for model import and visual validation on flat ground
- A simple fixed-direction locomotion objective for RSL-RL training:
  - target world `+X` velocity: `0.35 m/s`
  - penalties for lateral drift, yaw rate, tilt, action magnitude, and action rate
  - RSL-RL PPO config registered for `Ftr-Flat-Direct-v0`
- A mixed rough-terrain direct task registered as `Ftr-Rough-Direct-v0`:
  - terrain blocks are generated with Isaac Lab 4.5's `TerrainImporterCfg` / `TerrainGeneratorCfg`
  - terrain mix: flat ground, uphill/downhill stairs, uphill/downhill slopes
  - target world `+X` velocity: `0.30 m/s`
  - checkpoints are written under `logs/rsl_rl/ftr_rough_direct`
- A reference-inspired rough crossing task registered as `Ftr-ReferenceRough-Direct-v0`:
  - the environment supplies a fixed/random forward drive command
  - the policy controls only the four flippers
  - observation includes a 105-point forward height scan plus attitude, angular velocity, command, and flipper state
  - checkpoints are written under `logs/rsl_rl/ftr_reference_rough_direct`
- A migrated reference-terrain crossing task registered as `Ftr-ReferenceTerrain-Direct-v0`:
  - loads the reference project's USD terrain assets from `source/ftrbench/ftrbench/assets/terrain`
  - resets from `birth/*.json` start poses and trains toward the corresponding target points
  - velocity reward, success, and out-of-range checks follow the start-to-target path instead of fixed world `+X`
  - checkpoints are written under `logs/rsl_rl/ftr_reference_terrain_direct`

Recommended first validation flow:

```bash
# inside an Isaac Lab Python environment
python -m pip install -e source/ftrbench
python scripts/list_envs.py
python scripts/zero_agent.py --task Ftr-Flat-Direct-v0 --num_envs 1
python scripts/random_agent.py --task Ftr-Flat-Direct-v0 --num_envs 1
python scripts/rsl_rl/train.py --task Ftr-Flat-Direct-v0 --num_envs 64 --max_iterations 300 --headless
python scripts/rsl_rl/train.py --task Ftr-Rough-Direct-v0 --num_envs 64 --max_iterations 600 --headless
python scripts/rsl_rl/train.py --task Ftr-ReferenceRough-Direct-v0 --num_envs 64 --max_iterations 1000 --headless
python scripts/rsl_rl/train.py --task Ftr-ReferenceTerrain-Direct-v0 --num_envs 64 --max_iterations 3000 --headless
```

If you are not using a conda/venv Isaac Lab Python, replace `python` with your local `isaaclab.bat -p`.

The training run writes checkpoints and configuration snapshots under:

```bash
logs/rsl_rl/ftr_flat_direct
logs/rsl_rl/ftr_rough_direct
logs/rsl_rl/ftr_reference_rough_direct
logs/rsl_rl/ftr_reference_terrain_direct
```

To play the latest checkpoint after training:

```bash
python scripts/rsl_rl/play.py --task Ftr-Flat-Direct-v0 --num_envs 1
python scripts/rsl_rl/play.py --task Ftr-Rough-Direct-v0 --num_envs 1
python scripts/rsl_rl/play.py --task Ftr-ReferenceRough-Direct-v0 --num_envs 1
python scripts/rsl_rl/play.py --task Ftr-ReferenceTerrain-Direct-v0 --num_envs 1
```

## FTR Direct Training Objective

Both direct FTR tasks use the same action and observation layout. The action vector has 6 dimensions:

- `action[0]`: forward/backward track velocity command
- `action[1]`: yaw/turn command
- `action[2:6]`: four flipper joint position increments

The observation vector has 22 dimensions: base linear velocity, base angular velocity, projected gravity,
four flipper positions, target forward velocity, current velocity/yaw commands, and the previous action.

The reward is a weighted sum:

- positive reward for matching the target world `+X` velocity
- small positive reward for world `+X` progress
- penalties for lateral drift, yaw spinning, tilt, large actions, and fast action changes
- termination penalty if the robot base gets too low relative to the terrain origin or tilts too far

The rough task is intentionally conservative: it keeps the same 22-D observation instead of adding a height map.
This makes migration from flat training simple. If stair performance is poor, the next step is adding a
RayCaster height scan similar to the reference project's local height-map observation.

`Ftr-ReferenceRough-Direct-v0` follows that next step. It changes the learning problem from direct driving
to reference-style crossing:

- action space: 4-D flipper commands only
- observation space: 115-D, matching the reference crossing layout conceptually
- command: random fixed forward velocity in `0.15-0.30 m/s`
- reward: velocity tracking, upright/stability penalties, body-clearance penalty from the height scan, stuck penalty,
  terminal success bonus for crossing the terrain patch, and penalties for rollover or leaving the corridor

`Ftr-ReferenceTerrain-Direct-v0` keeps the same 4-D action and 115-D observation style, but replaces the procedural
rough terrain with the migrated reference terrain assets. The default terrain is `cur_steps_down`. Reset positions
come from the matching `birth/cur_steps_down.json` entries, so the robot starts on meaningful crossing lanes instead
of randomly landing on flat platform regions. The reward projects robot velocity onto the current start-to-target
direction, and success is reaching within `0.25 m` of the target point.

To train on another migrated reference terrain, override the terrain name through Hydra:

```bash
python scripts/rsl_rl/train.py --task Ftr-ReferenceTerrain-Direct-v0 --num_envs 64 --max_iterations 3000 --headless env.reference_terrain_name=cur_stairs_up
python scripts/rsl_rl/train.py --task Ftr-ReferenceTerrain-Direct-v0 --num_envs 64 --max_iterations 3000 --headless env.reference_terrain_name=cur_mixed
```

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/ftrbench

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: if the task naming convention changes, it may be necessary to update
        the search pattern in `scripts/list_envs.py`.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/ftrbench/ftrbench/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/ftrbench"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
