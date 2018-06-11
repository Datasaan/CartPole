# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:27:06 2017

@author: sanjeet
"""

#Import libraries
import gym
from gym import wrappers
import numpy as np
import keras


#
env = gym.make('CartPole-v1')
env = wrappers.Monitor(env, 'cartpole-experiment-1', force=True) # record
model=keras.models.load_model('model1.h5')

Rs = []
for _ in range(100):
    obs = env.reset()
    R = 0
    for _ in range(500):
        # render
        #env.render()
        # get action from DNN
        action = np.argmax(model.predict(np.reshape(np.array(obs),(1,4)))[0])
        obs, r, d, i = env.step(action)
        R += r
        if d:
            break
    Rs.append(R)
env.close()
# print result
print("Average Reward : {0}".format(sum(Rs)/len(Rs)))
#gym.upload('cartpole-experiment-1', api_key='sk_8BbS7bm5SGblFoPEEgdqg')