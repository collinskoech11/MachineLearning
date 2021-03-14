#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:09:07 2021

@author: collins
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('OptimalPolicy_angletol45.csv')
print(dataset.head())
print(dataset.shape)


X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values



np.random.seed(5)

n = 10
arms = np.random.rand(n)
eps = 0.1 #probability of exploration action

def reward(prob):
    reward = 0
    for i in range(10):
        if random.random() < prob:
            reward += 1
    return reward

#initialize memory array; has 1 row defaulted to random action index
av = np.array([np.random.randint(0,(n+1)), 0]).reshape(1,2) #av = action-value
#greedy method to select best arm based on memory array
def bestArm(a):
    bestArm = 0 #default to 0
    bestMean = 0
    for u in a:
        avg = np.mean(a[np.where(a[:,0] == u[0])][:, 1]) #calculate mean reward for each action
        if bestMean < avg:
            bestMean = avg
            bestArm = u[0]
    return bestArm

plt.xlabel("Number of times played")
plt.ylabel("Average Reward")
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
for i in range(500):
    if random.random() > eps: #greedy exploitation action
        choice = bestArm(av)
        thisAV = np.array([[choice, reward(arms[choice])]])
        av = np.concatenate((av, thisAV), axis=0)
        plt.scatter(X, y, color='red')
        plt.plot(X, lin_reg.predict(X), color='blue')
    else: #exploration action
        choice = np.where(arms == np.random.choice(arms))[0][0]
        thisAV = np.array([[choice, reward(arms[choice])]]) #choice, reward

av = np.concatenate((av, thisAV), axis=0) #add to our action-value memory array
    #calculate the mean reward
runningMean = np.mean(av[:,1])
print(plt.scatter(i, runningMean))