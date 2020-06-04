# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image 

from baseline import ReplayMemory
from baseline import Sample

class BasicSample():
  #define s,a,r,s',is_terminal of the MDP
  def __init__(self):
    self.state = np.zeros((84,84,4), dtype =np.unit8)
    self.action =0
    self.reward =0.0
    self.nextstate = np.zeros((84,84,4), dtype =np.unit8))
    self.is_terminal= False

  def assign(self,state,action,reward, nextstate,is_terminal):
    self.state[:]= state
    self.action =action
    self.reward =reward
    self.nextstate[:]=nextstate
    self.is_terminal = is_terminal

class BasicReplayMemory():

  def __init__(self, max_size):
    self.memory = [BasicSample() for _ in range(max_size)]
    self.max_size =max_size
    self.iter =0             #next element here
    self.cur_size=0

  def append(self, state,action,reward,nextstate,is_terminal):
    self.memory[self.iter].assign(state,action,reward,nextstate,is_terminal)
    self.iter +=1
    self.cur_size = min(self.cur_size+1 ,self.max_size)
    self.iter %=self.max_size

  def sample(self,batch_size):
    res = []
    for i ,j in enumerate(np.random.randint(0, self.cur_size, size=batch_size)):
      sample = self.memory[j]
      res.append((sample state,
                  samle.action,
                  sample.reward,
                  sample.nextstate,
                  sample.is_terminal))
    return res  

  def get_size(self):
    return self.cur_size
