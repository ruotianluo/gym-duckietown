#!/usr/bin/env python2

from __future__ import division, print_function

import numpy as np
import numpy
import gym

from gym_duckietown.envs import DuckietownEnv
import pyglet
import time

from scipy.misc import imsave

from transformations import euler_from_quaternion

import os

def main():

    env = gym.make('Duckietown-v0')
    env.reset()

    env.render()

    if not os.path.isdir('images'):
        os.mkdir('images')
    for i in range(11000):
        action = (np.random.random(2) -0.5) * 2
        if action[0] <0 and action[1]<0:
            action = -action

        obs, reward, done, info = env.step(action)
        state = env.stateData

        pos = state['position']
        angle = euler_from_quaternion(np.array(state['orientation']))
        print(pos)
        print(angle)

        if abs(1- state['position'][1]) < 0.25 or done:
            done = True

        print(action)
        print(obs.shape)
        # save images and ground truth
        imsave('images/'+str(pos[1])+','+str(angle[0])+'.jpg', obs.transpose())


        print('stepCount = %s, reward=%.3f' % (env.stepCount, reward))

        env.render()

        if True: #done:
            # print('done!')
            print('reset')
            env.reset()
            # env.render()

if __name__ == "__main__":
    main()
