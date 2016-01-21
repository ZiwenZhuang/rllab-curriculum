require_relative '../utils'

params = {
  mdp: {
    _name: "mujoco_1_22.hopper_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [32, 32],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "hopper",
  algo: {
    _name: "trpo",
    whole_paths: true,
    batch_size: 50000,
    max_path_length: 500,
    n_itr: 500,
    step_size: 0.01,
  },
  n_parallel: 4,
  snapshot_mode: "all",
  seed: 1,
}
command = to_command(params)
puts command
system(command)