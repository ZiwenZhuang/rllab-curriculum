require_relative './utils'

seed = 1

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
  },
  normalize_mdp: true,
  qf: {
    _name: "continuous_nn_q_function",
  },
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [32, 32],
    output_nl: "lasagne.nonlinearities.tanh",
  },
  exp_name: "svg0_box2d_cartpole",
  algo: {
    _name: "svg0",
    batch_size: 100,
    n_epochs: 100,
    epoch_length: 1000,
    min_pool_size: 10000,
    replay_pool_size: 100000,
    discount: 0.99,
    max_path_length: 100,
    eval_samples: 10000,
    eval_whole_paths: true,
    soft_target_tau: 0.001,
    policy_learning_rate: 1e-5,
  },
  snapshot_mode: "last",
  seed: seed,
}
command = to_command(params)
puts command
system(command)
