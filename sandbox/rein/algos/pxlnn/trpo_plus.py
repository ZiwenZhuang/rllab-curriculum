import tensorflow as tf
import time

from sandbox.rein.algos.pxlnn.plotter import Plotter
from sandbox.rein.algos.pxlnn.trpo import TRPO
from rllab.misc import special
import numpy as np
from rllab.misc import tensor_utils
import rllab.misc.logger as logger
from rllab.algos import util

# --
# Nonscientific printing of numpy arrays.
from sandbox.rein.algos.replay_pool import ReplayPool

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)


class TRPOPlus(TRPO):
    """TRPO+
    Modular extension to TRPO to allow for intrinsic reward.
    """

    def __init__(
            self,
            model=None,
            eta=0.1,
            model_pool_args=None,
            **kwargs):
        super(TRPOPlus, self).__init__(**kwargs)

        assert model is not None

        self._model = model
        self._eta = eta

        if model_pool_args is None:
            self._model_pool_args = dict(size=100000, min_size=32, batch_size=32)

        observation_dtype = "uint8"
        self._pool = ReplayPool(
            max_pool_size=self._model_pool_args['size'],
            observation_shape=(self.env.observation_space.flat_dim,),
            action_dim=self.env.action_dim,
            observation_dtype=observation_dtype,
            num_seq_frames=self._n_seq_frames,
            **self._model_pool_args
        )
        self._plotter = Plotter()

    def process_samples(self, itr, paths):
        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + self.discount * path_baselines[1:] - path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        if not self.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.center_adv:
                advantages = util.center_advantages(advantages)

            if self.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["ext_rewards"]) for path in paths]

            ent = np.mean(self.policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = np.array([tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])

            if self.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in actions])

            rewards = [path["rewards"] for path in paths]
            rewards = np.array([tensor_utils.pad_tensor(r, max_path_length) for r in rewards])

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = np.array([tensor_utils.pad_tensor(v, max_path_length) for v in valids])

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["ext_rewards"]) for path in paths]

            ent = np.sum(self.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        logger.log("Updating baseline ...")
        self.baseline.fit(paths)
        logger.log("Baseline updated.")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data

    def comp_int_rewards(self, paths):
        pass

    def fill_replay_pool(self, paths):
        logger.log('Filling replay pool ...')
        tot_path_len = 0
        for path in paths:
            path_len = len(path['rewards'])
            tot_path_len += path_len
            for i in range(path_len):
                obs = (path['observations'][i] * self._model.n_classes).astype(int)
                act = path['actions'][i]
                rew_orig = path['ext_rewards'][i]
                term = (i == path_len - 1)
                self._pool.add_sample(obs, act, rew_orig, term)
        logger.log('{} samples added to replay pool ({}).'.format(tot_path_len, self._pool.size))

    def encode_obs(self, obs):
        """
        Observation into uint8 encoding, also functions as target format
        """
        assert np.max(obs) <= 1.0
        return (obs * self._model.n_classes).astype("uint8")

    def decode_obs(self, obs):
        """
        From uint8 encoding to original observation format.
        """
        return obs / float(self._model.num_classes)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.start_worker()
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                with logger.prefix('itr #%d | ' % itr):

                    # --
                    # Sample trajectories.
                    paths = self.obtain_samples(itr)

                    # --
                    # Save external rewards.
                    for path in paths:
                        path['ext_rewards'] = np.array(path['rewards'])

                    # --
                    # Encode all observations into uint8 format
                    for path in paths:
                        path['observations'] = self.encode_obs(path['observations'])

                    # --
                    # Fill replay pool.
                    self.fill_replay_pool(paths)

                    # --
                    # Compute intrinisc rewards.
                    self.comp_int_rewards(paths)

                    # --
                    # Compute deltas, advantages, etc.
                    samples_data = self.process_samples(itr, paths)

                    # --
                    self.optimize_policy(itr, samples_data)

                    # --
                    # Diagnostics
                    self.log_diagnostics(paths)
                    logger.log("Saving snapshot ...")
                    params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = samples_data["paths"]
                    # FIXME: bugged
                    # logger.save_itr_params(itr, params)
                    logger.log("saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")
        self.shutdown_worker()