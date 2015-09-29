import numpy as np
import sys
from misc.console import Message, prefix_log
from misc.tensor_utils import flatten_tensors
import multiprocessing
import theano
import theano.tensor as T
import pydoc
from remote_sampler import RemoteSampler


def reduce_add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def reduce_mul(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a * b


# Unconstrained TRPO
class UTRPO(object):

    def __init__(
            self, n_itr=500, max_samples_per_itr=100000,
            max_steps_per_itr=np.inf, discount=0.99, stepsize=0.015,
            initial_lambda=1, max_opt_itr=100,
            n_parallel=multiprocessing.cpu_count(), adapt_lambda=True,
            reuse_lambda=True, sampler_module='algo.rollout_sampler',
            optimizer_module='scipy.optimize.fmin_l_bfgs_b'):
        self._n_itr = n_itr
        self._max_samples_per_itr = max_samples_per_itr
        self._max_steps_per_itr = max_steps_per_itr
        self._discount = discount
        self._stepsize = stepsize
        self._initial_lambda = initial_lambda
        self._max_opt_itr = max_opt_itr
        self._adapt_lambda = adapt_lambda
        self._n_parallel = n_parallel
        # whether to start from the currently adapted lambda on the next
        # iteration
        self._reuse_lambda = reuse_lambda
        self._sampler_module = sampler_module
        self._optimizer_module = optimizer_module

    def _new_surrogate_obj(
            self, policy, input_var, Q_est_var, pi_old_vars, action_vars,
            lambda_var):
        probs_vars = policy.probs_vars
        N_var = input_var.shape[0]
        mean_kl = None
        lr = None
        for probs_var, pi_old_var, action_var in zip(
                probs_vars, pi_old_vars, action_vars):
            mean_kl = reduce_add(mean_kl, T.mean(
                T.sum(
                    pi_old_var * (
                        T.log(pi_old_var + 1e-6) - T.log(probs_var + 1e-6)
                    ),
                    axis=1)
                ))
            pi_old_selected = pi_old_var[T.arange(N_var), action_var]
            pi_selected = probs_var[T.arange(N_var), action_var]
            lr = reduce_mul(lr, pi_selected / (pi_old_selected + 1e-6))
        # formulate as a minimization problem
        surrogate_loss = - T.mean(lr * Q_est_var)
        surrogate_obj = surrogate_loss + lambda_var * mean_kl
        return surrogate_obj, surrogate_loss, mean_kl

    def train(self, gen_mdp, gen_policy):

        mdp = gen_mdp()
        input_var = T.matrix('input')  # N*Ds
        Q_est_var = T.vector('Q_est')  # N
        action_range = range(len(mdp.action_dims))
        pi_old_vars = [T.matrix('pi_old_%d' % i) for i in action_range]
        action_vars = [T.vector('action_%d' % i, dtype='uint8')
                       for i in action_range]
        lambda_var = T.scalar('lambda')

        policy = gen_policy(mdp.observation_shape, mdp.action_dims, input_var)
        surrogate_obj, surrogate_loss, mean_kl = \
            self._new_surrogate_obj(
                policy, input_var, Q_est_var, pi_old_vars, action_vars,
                lambda_var)

        grads = theano.gradient.grad(surrogate_obj, policy.params)

        all_inputs = [input_var, Q_est_var] + pi_old_vars + action_vars + \
            [lambda_var]

        with Message("Compiling functions..."):
            compute_surrogate_obj = theano.function(
                all_inputs, surrogate_obj, on_unused_input='ignore',
                allow_input_downcast=True
                )
            compute_mean_kl = theano.function(
                all_inputs, mean_kl, on_unused_input='ignore',
                allow_input_downcast=True
                )
            compute_grads = theano.function(
                all_inputs, grads, on_unused_input='ignore',
                allow_input_downcast=True
                )

        lambda_ = self._initial_lambda

        optimizer = pydoc.locate(self._optimizer_module)

        with RemoteSampler(
                self._sampler_module, self._n_parallel, gen_mdp,
                gen_policy) as sampler:

            for itr in xrange(self._n_itr):

                itr_log = prefix_log('itr #%d | ' % (itr + 1))

                cur_params = policy.get_param_values()

                itr_log('collecting samples...')

                tot_rewards, n_traj, all_obs, Q_est, all_pi_old, all_actions = \
                    sampler.request_samples(
                        itr, cur_params, self._max_samples_per_itr,
                        self._max_steps_per_itr, self._discount)

                all_input_values = [all_obs, Q_est] + all_pi_old + all_actions

                def evaluate_cost(lambda_):
                    def evaluate(params):
                        policy.set_param_values(params)
                        inputs_with_lambda = all_input_values + [lambda_]
                        val = compute_surrogate_obj(*inputs_with_lambda)
                        return val.astype(np.float64)
                    return evaluate

                def evaluate_grad(lambda_):
                    def evaluate(params):
                        policy.set_param_values(params)
                        grad = compute_grads(*(all_input_values + [lambda_]))
                        flattened_grad = flatten_tensors(map(np.asarray, grad))
                        return flattened_grad.astype(np.float64)
                    return evaluate

                avg_reward = tot_rewards * 1.0 / n_traj

                itr_log('avg reward: %.3f over %d trajectories' %
                        (avg_reward, n_traj))

                if not self._reuse_lambda:
                    lambda_ = self._initial_lambda

                with Message('trying lambda=%.3f...' % lambda_):
                    result = optimizer(
                        func=evaluate_cost(lambda_), x0=cur_params,
                        fprime=evaluate_grad(lambda_),
                        maxiter=self._max_opt_itr
                        )
                    mean_kl = compute_mean_kl(*(all_input_values + [lambda_]))
                    itr_log('lambda %.3f => mean kl %.3f' % (lambda_, mean_kl))
                # do line search on lambda
                if self._adapt_lambda:
                    max_search = 4
                    if itr < 2:
                        max_search = 10
                    if mean_kl > self._stepsize:
                        for _ in xrange(max_search):
                            lambda_ = lambda_ * 2
                            with Message('trying lambda=%.3f...' % lambda_):
                                result = optimizer(
                                    func=evaluate_cost(lambda_), x0=cur_params,
                                    fprime=evaluate_grad(lambda_),
                                    maxiter=self._max_opt_itr)
                                inputs_with_lambda = all_input_values + [lambda_]
                                mean_kl = compute_mean_kl(*inputs_with_lambda)
                                itr_log('lambda %.3f => mean kl %.3f' % (lambda_, mean_kl))
                            if np.isnan(mean_kl):
                                import ipdb
                                ipdb.set_trace()
                            if mean_kl <= self._stepsize:
                                break
                    else:
                        for _ in xrange(max_search):
                            try_lambda_ = lambda_ * 0.5
                            with Message('trying lambda=%.3f...' % try_lambda_):
                                try_result = optimizer(
                                    func=evaluate_cost(try_lambda_), x0=cur_params,
                                    fprime=evaluate_grad(try_lambda_),
                                    maxiter=self._max_opt_itr)
                                inputs_with_lambda = all_input_values + [lambda_]
                                try_mean_kl = compute_mean_kl(*inputs_with_lambda)
                                itr_log('lambda=%.3f => mean kl %.3f' % (try_lambda_, try_mean_kl))
                            if np.isnan(mean_kl):
                                import ipdb
                                ipdb.set_trace()
                            if try_mean_kl > self._stepsize:
                                break
                            result = try_result
                            lambda_ = try_lambda_
                            mean_kl = try_mean_kl

                result_x, result_f, result_d = result
                itr_log('optimization finished. new loss value: %.3f. mean kl: %.3f' % (result_f, mean_kl))
                sys.stdout.flush()