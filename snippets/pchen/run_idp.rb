require_relative '../rocky/utils'

quantile = 1
seed = 1

params = {
  mdp: {
    # _name: "box2d.inverted_double_pendulum_mdp",
    _name: "mujoco_1_22.inverted_double_pendulum_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    # hidden_sizes: [],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "trpo_idp_seed_#{seed}",
  algo: {
    #_name: "ppo",
    #step_size: 0.05,
    #binary_search_penalty: false,

    _name: "trpo",
    step_size: 0.1,
    backtrack_ratio: 0.7,
    max_backtracks: 15,
    cg_iters: 10,
    batch_size: 10000,


    # _name: "cem",
    # extra_decay_time: 300,

    whole_paths: true,
    max_path_length: 200,
    n_itr: 5000,
    plot: true,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  seed: seed,
  plot: true,
}
command = to_command(params)
puts command
system(command)
