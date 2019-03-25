#! python3

from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import tf.random as rand

class PointEnv(Env):
    ''' A learning practice for running the library demo
    '''
    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        reward = - (x**2 + y**2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print ('current state:', self._state)

class LogisticEnv(Env):
    ''' Trying to learn to optimize logistic regression problem
    '''
    def __init__(self, sess= None):
        # a dictionary specifying configurations
        # Don't put object in here
        self.configs = {
            "num_data": 10,
            # assuming the value is x, the number of parameters in w and d will be x^2 + x
            "x_dim": 3,
            "lambda": tf.constant(0.0005), # for l-2 regularization according to the paper
            "sample_first": True, # whether to sample the x-s first.
        }
        
        # data generator of 2 multivariate Gaussians
        # (zero-th one for y == 0, another for y == 1)
        self.distributions = [self._rand_Gaussian_Dist(), self._rand_Gaussian_Dist()]

        # assign tf session
        if sess != None:
            self.sess = sess
        else:
            self.sess = tf.Session()

        # setup formula
        self.vars = {}
        with tf.variable_scope("discriminator") as scope:
            self.vars["w"] = tf.Variable(name= "w", shape= (self.configs["x_dim"],1))
            self.vars["b"] = tf.Variable(name= "b", shape= (1,1))
            # where 'None' indicates the number of data generated
            self.vars["x"] = tf.placeholder(tf.float32, shape= (self.configs["x_dim"], None), name= "x")
            self.vars["y"] = tf.placeholder(tf.float32, shape= (1, None), name= "y") # the true value

            self.vars["out"] = tf.math.sigmoid((tf.matmul(w, x, transpose_a= True) + b), name= "logistic_out")

        with tf.variable_scope("loss_func") as scope:
            self.vars["losses"] = (self.vars["y"] * tf.math.log(self.vars["out"]) + \
                    (1 - self.vars["y"]) * tf.math.log(1 - self.vars["out"]))
            self.vars["loss"] = tf.reduce_mean(self.vars["losses"], 1) + self.configs["lambda"] / 2 * tf.norm(self.vars["w"], ord= 2)

        # sample data from each distributions
        if self.configs["sample_first"]:
            
        
    def _rand_Gaussian_Dist(self):
        ''' The method generate a multivariate gaussian distribution with 
            random mean and covariance.
            It returns a distribution that can be sampled
        '''
        mean = rand.uniform(shape= [self.configs["x_dim"]])
        covar = rand.uniform(shape= [self.configs["x_dim"], self.configs["x_dim"]], maxval= 1)

        # generate the distribution
        return tfd.MultivariateNormalFullCovariance(
                loc= meam,
                covariance_matrix= covar
            )

    @property
    def observation_space(self):
        # the weights of thelogistics objective equation
        

