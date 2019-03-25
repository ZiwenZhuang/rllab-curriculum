#! python3

# add library to python path if needed
import sys
sys.path.append('/home/leo/Projects/rllab-curriculum')

import argparse
import rllab.msic.logger as logger
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from logisticEnv import PointEnv

def main(args):
    # set model parameters saving path directly to the logger
    if args.save_path:
        logger.set_snapshot_dir(args.save_path)

    env = normalize(PointEnv())
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
    )
    algo.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do experiment on learning to optimize")
    # add arguments for command line
    parser.add_argument("--save", dest="save_path", action="store", nargs=1, type=str, default=None, \
        help="The relative path to store the model parameters.")


    args = parser.parse_args()
    main(args)
