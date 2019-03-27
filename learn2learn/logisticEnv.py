#! python3

from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
from rllab.misc import logger

import numpy as np
import pickle
import os

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
    def __init__(self, sess= None, data_path= None, weights_path= None):
        ''' Both paths are file path, not directory path. (they has to be pkl file stored previously)
        '''
        # a dictionary specifying configurations
        # Don't put object in here
        self.configs = {
            # each type of eample will be drawn half of the total sample amount.
            "num_data_total": 100,
            # assuming the value is x, the number of parameters in w and d will be x + 1
            "x_dim": 3,
            "lambda": tf.constant(0.0005), # for l-2 regularization according to the paper
            "data_path": data_path, # load and save (X,Y) data
            "weights_path": weights_path, # load and save learnt weights
        }

        logger.log("Logistic regression environment initializing...")
        
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

        # if nowhere to load the data, sample them
        if self.configs["data_path"] is None or not os.path.isfile(self.configs["data_path"]):
            # both of which will be concatencate into a matrix with (n) columns
            X, Y = self._generate_data(self.configs["num_data_total"])
            self.data = (X, Y)
            logger.log("(X,Y) data generated.")
            # check again whether to store the data
            if isinstance(self.configs["data_path"], str):
                pickle.dump(self.data)
                logger.log("(X,Y) data stored at: %s" % self.configs["data_path"])
        else:
            with open(self.configs["data_path"]) as f:
                self.data = pickle.load(f)
                logger.log("(X,Y) data loaded at: %s" % self.configs["data_path"])

        # initialize the weights
        # TODO: provide method to load and store weights at certain frequencies.
        self.sess.run([self.vars["w"].initializer, self.vars["b"].initializer])
        logger.log("Logistic regression initialization done.")
            
    def _generate_data(n):
        ''' Generate 'n' number of data (with input and output) in total.
            Each type of data is evenly generated in terms of amount.
            return: (X, Y)
                X: a (d by n) np array
                Y: a (1 by n) np array
        '''
        X = []
        Y = []
        for i, dist in enumerate(self.distributions):
            num = n // len(self.distributions)
            X_data = self.sess.run(dist.sample(num)) # calculate the data
            X.append(X_data.transpose()) # transpose the array, so that each column is a vector
            Y.append(numpy.array([[i for _ in range(num)]]))
        # concatencate each array into matrix.
        X = np.concatenate(X, axis= 1)
        Y = np.concatenate(Y, axis= 1)
        return (X, Y)
        
    def _rand_Gaussian_Dist(self):
        ''' The method generate a multivariate gaussian distribution with 
            random mean and covariance.
            It returns a distribution that can be sampled
        '''
        mean = rand.uniform(shape= [self.configs["x_dim"]])
        covar = rand.uniform(shape= [self.configs["x_dim"], self.configs["x_dim"]], maxval= 1)
        covar = numpy.dot(covar, covar.transpose()) #ensure that the random generated matrix is positive-semidefinite

        # generate the distribution
        return tfd.MultivariateNormalFullCovariance(
                loc= meam,
                covariance_matrix= covar
            )

    @property
    def observation_space(self):
        # the weights of thelogistics objective equation
        

