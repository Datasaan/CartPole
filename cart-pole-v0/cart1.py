# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 19:11:15 2017

@author: sanjeet
"""


# Import Libraries
import gym
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
import numpy as np


# Create Environment variable 
env = gym.make('CartPole-v0')


# Create the agent learner model
model=Sequential()
model.add(Dense(32,input_dim=4,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32,input_dim=4,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16,input_dim=4,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam',loss='mse',metrics=['mse'])


# Training the agent 
for _ in range(4000):
    print _
    observation = env.reset()

    # gather data to train a model
    actions = []
    observations = []

    # total reward
    R = 0
    
    for _ in range(200):
        action = env.action_space.sample()
        
        # save the observation and the action
        observations.append(observation)
        actions.append([action ^ 0b1, action & 0b1])

        # and take an action
        observation, reward, done, info = env.step(action)
        
        # sum of rewards
        R += reward

        if done:
            if R >= 50:
                # train
                model.fit(np.array(observations),np.array(actions),verbose=0)
            break

model.save('model1.h5')