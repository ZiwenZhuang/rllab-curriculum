# try eps sampling to see if it can make Q learning matches a3c paper more

# in addition try lr sampling & sharing opt states or not

import logging
import os,sys
import numpy as np
import itertools

from rllab.envs.sliding_mem_env import SlidingMemEnv
from sandbox.pchen.async_rl.async_rl.agents.boltzmann_dqn_agent import BoltzmannDQNAgent
from sandbox.pchen.dqn.envs.atari import AtariEnvCX

sys.path.append('.')

from sandbox.pchen.async_rl.async_rl.agents.a3c_agent import A3CAgent
from sandbox.pchen.async_rl.async_rl.agents.dqn_agent import DQNAgent, Bellman
from sandbox.pchen.async_rl.async_rl.algos.a3c_ale import A3CALE
from sandbox.pchen.async_rl.async_rl.algos.dqn_ale import DQNALE
from sandbox.pchen.async_rl.async_rl.utils.get_time_stamp import get_time_stamp
from sandbox.pchen.async_rl.async_rl.utils.ec2_instance import instance_info, subnet_info
from sandbox.pchen.async_rl.async_rl.bonus_evaluators.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.pchen.async_rl.async_rl.preprocessor.image_vectorize_preprocessor import ImageVectorizePreprocessor
from sandbox.pchen.async_rl.async_rl.hash.sim_hash import SimHash

from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, stub
from rllab import config

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

class VG(VariantGenerator):
    @variant
    def seed(self, lr):
        return [np.random.randint(10000), ]

    @variant
    def total_t(self):
        # return [2*7 * 3*10**6]
        # half time, short trial
        return [35 * 10**6]

    @variant
    def n_processes(self):
        return [18, ]

    # @variant
    # def entropy_bonus(self):
    #     return [0.01]

    @variant
    def target_update_frequency(self):
        yield 40000

    @variant
    def eps_test(self):
        yield 0.01

    @variant
    def eval_frequency(self, target_update_frequency):
        # yield target_update_frequency * 1
        yield 10**6

    @variant
    def game(self, ):
        return ["pong", "breakout", "space_invaders", ]
        # return ["space_invaders"]
        # return ["breakout"]
        # return ["seaquest", "enduro", "breakout",]

    @variant
    def n_step(self, ):
        return [5,]

    @variant
    def share_model(self, n_step):
        return [n_step == 1]

    @variant
    def bellman(self, ):
        # return ["q"]
        return [
            Bellman.q,
            # Bellman.sarsa,
        ]

    @variant
    def lr(self, game):
        return [
            np.random.uniform(-4, -2)
            for _ in range(4)
        ]

    @variant
    def random_seed(self, ):
        return [True, ]

    @variant
    def manual_start(self, ):
        return [True, ]

    @variant
    def adaptive_entropy(self, ):
        return [True, ]

    @variant
    def temp_init(self, ):
        return [1e-2, ]

    @variant
    def opt_share(self, ):
        return [True, False]

vg = VG()
variants = vg.variants(randomized=False)

print(len(variants))

for v in variants[:]:
    locals().update(v)

    # Problem setting
    eval_n_runs = 30

    # The meat ---------------------------------------------
    # env = AtariEnv(
    #     rom_filename=os.path.join(rom_dir,game+".bin"),
    #     plot=plot,
    #     record_ram=record_ram,
    # )
    env = AtariEnvCX(
        game=game,
        obs_type="image",
        life_terminating=True,
        color_averaging=False,
        random_seed=random_seed,
        color_max=True,
        initial_manual_activation=manual_start,
    )
    env = SlidingMemEnv(env)
    test_env = SlidingMemEnv(AtariEnvCX(
        game=game,
        obs_type="image",
        life_terminating=False,
        color_averaging=False,
        random_seed=random_seed,
        color_max=True,
        initial_manual_activation=manual_start,
    ))

    agent = DQNAgent(
        # n_actions=env.number_of_actions,
        env=env,
        target_update_frequency=target_update_frequency,
        eps_test=eps_test,
        t_max=n_step,
        share_model=share_model,
        optimizer_args=dict(
            lr=lr,
            eps=1e-5,
            alpha=0.9,
        ),
        bellman=bellman,
        # adaptive_entropy=adaptive_entropy,
        # temp_init=temp_init,
        sample_eps=True,
        share_optimizer_states=opt_share,
    )
    algo = DQNALE(
        total_steps=total_t,
        n_processes=n_processes,
        env=env,
        test_env=test_env,
        agent=agent,
        eval_frequency=eval_frequency,
        eval_n_runs=eval_n_runs,
    )

    # sys stuff
    comp_cores = max(int(20 / n_processes), 1)
    config.ENV = dict(
        MKL_NUM_THREADS=comp_cores,
        NUMEXPR_NUM_THREADS=comp_cores,
        OMP_NUM_THREADS=comp_cores,
    )
    # config.AWS_INSTANCE_TYPE = "c4.8xlarge"
    config.AWS_INSTANCE_TYPE = "m4.10xlarge"
    config.AWS_SPOT = True
    config.AWS_SPOT_PRICE = '1.'
    config.AWS_REGION_NAME = 'us-west-1'
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]


    run_experiment_lite(
        algo.train(),
        exp_prefix="1020_async_q_opt_share",
        seed=v["seed"],
        variant=v,
        mode="ec2",
        # terminate_machine=False,
        # mode="local_docker",
        # mode="local",
        #

        # mode="lab_kube",
        # n_parallel=0,
        # use_gpu=False,
        # node_selector={
        #     "aws/type": "c4.8xlarge",
        #     "openai/computing": "true",
        # },
        # resources=dict(
        #     requests=dict(
        #         cpu=17.1,
        #     ),
        #     limits=dict(
        #         cpu=17.1,
        #     )
        # )
    )
