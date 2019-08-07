#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:28:50 2019

@author: maylibooyah69
"""
import numpy as np

class RL_env():
    def __init__(self,epoched_df):
        '''takes in the data of a rat over one single trial'''
        self.epoched_df=epoched_df
        self.Q=epoched_df['Q']
        self.reward=epoched_df['reward']
        self.test_alphaL=epoched_df['alpha_loss'].iloc[0] # stored as scaler
        self.test_alphaG=epoched_df['alpha_gain'].iloc[0]
        self.test_beta=epoched_df['beta'].iloc[0]
        self.PE=epoched_df['PE']
        self.count=0 # counting from 0
    def step(self):
        temp=self.reward.iloc[self.count]+0
        self.count+=1
        return temp
    def get_switchid(self):
        pass
    def init_Q(self):
        left=self.epoched_df[self.epoched_df['action']==1]['Q'].iloc[0]
        right=self.epoched_df[self.epoched_df['action']==2]['Q'].iloc[0]
        return np.array([right,left])

class Rat():
    def __init__(self,epoched_df,alphaG=None,alphaL=None,beta=None,gamma=0,init_Q=np.array([-1,-1])):
        self.gamma=gamma
        self.df=epoched_df
        self.count=0
        self.PE=0 # prediction error (Q-R)
        if alphaG==None and alphaL==None and beta==None:
            self.alphaG=epoched_df['alpha_gain'].iloc[1]
            self.alphaL=epoched_df['alpha_loss'].iloc[1]
            self.beta=epoched_df['beta'].iloc[1]
            self.actions=epoched_df['action']
        else:
            self.alphaG=alphaG
            self.alphaL=alphaL
            self.beta=beta
        if (init_Q==np.array([-1,-1])).all():
            self.Q=np.random.rand(2) # Q[0] represent left
        else:
            self.Q=init_Q
    def get_action(self):
        temp=self.count+0
        self.count+=1
        return self.actions.iloc[temp]
    def rat_chose(self):
        if np.random.uniform()<1/(1+np.exp(-self.Q)): # Q represent the difference between going left - right
            return 1
        else:
            return 2
    def update(self,obs): # 1 represent left 
        action_id=int(self.get_action()-1)
        if int(obs)==1: # alpha_gain
            self.Q[action_id]=(1-self.alphaG)*self.Q[action_id]+self.alphaG*(obs+self.gamma*np.max(self.Q))
        elif int(obs)==0: # alpha_loss
            self.Q[action_id]=(1-self.alphaL)*self.Q[action_id]+self.alphaL*(obs+self.gamma*np.max(self.Q))
        else:
            print('error')
        return self.Q[action_id]
    
    
def train_rat(env,rat,it_num):
    QLog=rat.Q
    qlog=[0]
    for i in range(it_num):
        obs=env.step()
        q=rat.update(obs)
        QLog=np.vstack((QLog,rat.Q))
        qlog.append(q)
    return QLog,qlog

rat=Rat(None,alphaG=0.7,alphaL=0.2,beta=4)
print(rat.Q)