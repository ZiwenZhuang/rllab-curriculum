from algo.ilqg import ILQG
from mdp.cartpole_mdp import CartpoleMDP

mdp = CartpoleMDP()
mdp.horizon = 70
ILQG().train(mdp)