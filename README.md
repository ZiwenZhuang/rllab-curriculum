# rllab-learn2learn

This repo is forked from https://github.com/florensacc/rllab-curriculum which possibly from rllab.

## Installation

We have only installed it through anaconda, which is the only way we could provide. Unfortunately, the initial `rllab` is deprecated. And for the safety and easy to locate problem while installing, please install `mujoco-py` manually in the virtual environment.

```bash
cd path/to/rllab-learn2learn
conda env create -f environment.yml

# manually install mujoco-py in the virtual environment
# manually upgrade tensorflow to meet the dependency requirement of tensorflow-probability
```

### Error & Solution

- If you cannot use GPU version of tensorflow, please change `tensorflow-gpu` to `tensorlfow` in environment.yml
- If you are going to use mujoco environment, which is commercial, you need to acquire a key and install it. (please notice the mujoco version needed by mujoco-py)
- If you see `file not found` error with `patchelf` prompted, please install it manually with `sudo apt install patchelf`
- If you get error on incompatible ipython dependency, please manually upgrade ipython via pip. (in the virtual environment)
- If you see other dependencies that are not solved by conda itself, just install the version that meets the requirement. (there should not be too much)

You can read more to setup `rllab` and see documentation at [https://rllab.readthedocs.org/en/latest/](https://rllab.readthedocs.org/en/latest/).

## Changes

All experiments code are at `learn2learn` directory. Other from that, they are inherited from upstream.
