#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:04:00 2019

@author: maylibooyah69
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize

left = 0.2 # probablity of reward in left
right = 1 - left # probability of reward in right

# sigmoid function for nd; return the value of sigmoid with argument beta
# q can be n dimensional, either list or int/float
# beta should be between 0 and 1; determine the sharpness of sigmoid
# we can also try different functions here
def sigmoid(beta,q):
    q=np.array(q)
    val = 1/(1+np.exp(0-beta*q))
    return val
def ratchose(Q): # 0 means left
    if random.uniform(0,1)<sigmoid(beta,Q[0]-Q[1]):
        return np.array([1,0])
    else:
        return np.array([0,1]) 
    
# parameter: action, a list of numpy arrays of action data; reward, a list of np array of reward data
# beta, sharpness of sigmoid; alpha, learning rate
# return the sum of log-likelihood
def neg_log_likelihood(alpha,beta,actions,rewards,Q=[0,0],gamma=0): 
    n = len(actions)
    sum_ll = 0
    for i in range(n):
        turn = actions[i]
        rew = rewards[i]
        Q = Q + alpha*turn*(rew + gamma*np.max(Q)-Q)
        dQ = Q[0] - Q[1]
        if np.array_equal(turn,np.array([1,0])):
            prob = 1/(np.exp(0-beta*dQ)+1)
        else:
            prob = 1 - 1/(np.exp(0-beta*dQ)+1)
        
        sum_ll = sum_ll - np.log(prob + np.exp(0-8)) # add a smoother to avoid warnings
    
    return sum_ll

# params = [alpha0,beta0]
# args = [actions,rewards]
def helper_func(params,args):
    alpha0 = params[0]
    beta0 = params[1]
    actions = args[0]
    rewards = args[1]
    
    sum_ll = neg_log_likelihood(alpha0,beta0,actions,rewards)
    
    return sum_ll

class toyQ_2choice:
    '''a unchanging env with probability of giving one
    of two rewards based on params to two different choices'''
    obs_size=1
    action_size=2
    def __init__(self,left=0.2,right=0.8,state=0):
        self.left=left
        self.right=right
        self.obslog=[]
        self.state=state
    def step(self,action,state=0):
        '''takes in the action param as a int of 0 or 1
        where 0 is go left and 1 is to right, and returns
        a return reward where 0 is no reward'''
        reward_site = random.uniform(0,1)
        if reward_site < self.right and np.array_equal(action,np.array([0,1])):
            #if the rat goes right and the reward is on the right
            obs=np.array([0,1])
        elif reward_site >= self.right and np.array_equal(action,np.array([1,0])):
            obs=np.array([1,0])
        else:
            obs=np.array([0,0])
        self.obslog.append(obs)
        return obs
    
class WSLS_rat: # to-do
    pass

class sig_rat:
    def __init__(self,env,alpha=0.2,beta=4,gamma=0):
        # alpha-learning rate, beta-sigmoid slop, gamma-discount factor of future reward
        '''Takes in the beta, gamma ,and the 
        environment of the rat'''
        self.lhLog=[]
        self.choiceLog=[]
        self.beta=beta
        self.gamma=gamma
        self.env=env
        self.alpha=alpha
        self.action_size=env.action_size
        self.obs_size=env.obs_size
        self.Q=np.zeros((self.obs_size,)+(self.action_size,)) # Q-table starts from 0
        self.Qlog=np.zeros((self.obs_size,)+(self.action_size,))
    def get_choice(self): # 0 means left
        lh=sigmoid(self.beta,self.Q[self.env.state][0]-self.Q[self.env.state][1])
        self.lhLog.append(lh)
        if random.uniform(0,1)<lh:
            self.choice=np.array([1,0])
        else:
            self.choice=np.array([0,1])
        self.choiceLog.append(self.choice)
        return self.choice
    def update(self,obs):
        '''Takes obs/reward and update its Q-table'''
        self.Q[self.env.state] = self.Q[self.env.state] + self.alpha*self.choice*\
        (obs + self.gamma*np.max(self.Q[self.env.state])-self.Q[self.env.state])
        self.Qlog=np.vstack((self.Qlog,self.Q))
        return self.Q[self.env.state]
    
class FSrat:
    def __init__(self,env,alphaF=0.2,alphaS=0.1,beta=4,gamma=0): # fail and success
        # alpha-learning rate, beta-sigmoid slop, gamma-discount factor of future reward
        '''Takes in the beta, gamma ,and the 
        environment of the rat'''
        self.lhLog=[]
        self.choiceLog=[]
        self.beta=beta
        self.env=env
        self.alphaF=alphaF
        self.gamma=gamma
        self.alphaS=alphaS
        self.action_size=env.action_size
        self.obs_size=env.obs_size
        self.Q=np.zeros((self.obs_size,)+(self.action_size,)) # Q-table starts from 0
        self.Qlog=np.zeros((self.obs_size,)+(self.action_size,))
    def get_choice(self): #[1,0] means left
        lh=sigmoid(self.beta,self.Q[self.env.state][0]-self.Q[self.env.state][1])
        self.lhLog.append(lh)
        if random.uniform(0,1)<lh:
            self.choice=np.array([1,0])
        else:
            self.choice=np.array([0,1])
        self.choiceLog.append(self.choice)
        return self.choice
    def update(self,obs):
        '''Takes obs/reward and update its Q-table'''
        used_alpha=[self.alphaF,self.alphaS][max(obs)]
        self.Q[self.env.state] = self.Q[self.env.state] + used_alpha*self.choice*\
        (obs + self.gamma*np.max(self.Q[self.env.state])-self.Q[self.env.state])
        self.Qlog=np.vstack((self.Qlog,self.Q))
        return self.Q[self.env.state]
    
    pass

class SQFSrat: # Single Q Failure-Success rat
    def __init__(self,env,alphaF=0.2,alphaS=0.1,beta=4,gamma=0):
        # alpha-learning rate, beta-sigmoid slop, gamma-discount factor of future reward
        '''Takes in the beta, gamma ,and the 
        environment of the rat'''
        self.lhLog=[]
        self.choiceLog=[]
        self.beta=beta
        self.gamma=gamma
        self.env=env
        self.alpha=alpha
        self.action_size=env.action_size
        self.obs_size=env.obs_size
        self.Q=0 # Q starts from 0
        self.Qlog=[0]
    def get_choice(self): # 0 means left
        lh=sigmoid(self.beta,self.Q)
        self.lhLog.append(lh)
        if random.uniform(0,1)<lh:
            self.choice=np.array([1,0])
        else:
            self.choice=np.array([0,1])
        self.choiceLog.append(self.choice)
        return self.choice
    def update(self,obs):
        '''Assumes that going going left and get no reward is the
        same as going right and get an reward, and going right and
        not getting a reward is the same as going left and not getting'''
        if (obs==np.array([1,0])).all() and (self.choice==np.array([1,0])).all() or\
        (obs==np.array([0,0])).all() and (self.choice==np.array([0,1])).all():
            self.Q=(1-self.alpha)*self.Q+self.alpha*1 # going left and getting an reward of one
            self.Q=(1-self.alpha)*self.Q+self.alpha*1
        elif ((obs==np.array([0,0])).all() and (self.choice==np.array([1,0]))).all() or\
        (obs==np.array([0,1])).all() and (self.choice==np.array([0,1])).all():
            self.Q=(1-self.alpha)*self.Q # going left and not getting an reward
            self.Qlog.append(self.Q)
        return self.Q    
    
    
def train_rat(env,rat,it_num,every=500):
    for i in range(it_num):
        action=rat.get_choice()
        obs=env.step(action)
        rat.update(obs)
        if i%every==-1:
            print(np.mean(rat.Qlog,axis=0))
    return env,rat
