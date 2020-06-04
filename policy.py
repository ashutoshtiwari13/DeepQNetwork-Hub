# -*- coding: utf-8 -*-

import numpy as np
import attr

class Policy:
  """
  An MDP policy . A policy is an engine which takes inputs as 'states' and returns a chosen 'actions'

  """
  def select_action(self,**kwargs):
    """
    Used by the agents to select an action

    Returns
    --------------
       An object representing the chosen action
    """
    raise NotImplementedError('This needs to be overidden')

class UniformRandomPolicy(Policy):
  """
  Choses a discreate action with uniform random policy

   Parameters
   -----------
   num_actions:int
      Number of actions to choose from , which should be >0 , Raises error if <=0
  """
  def __init__(self,num_actions):
    assert num_actions>=1
    self.num_actions=num_actions

  def select_action(self,**kwargs):
    return np.random.randint(0, self.num_actions)

  def get_config(self):
    return {'num_actions': self.num_actions}

class GreedyPolicy(Policy):
  """
  Returns the best action or 'greedy' action according to greedy values.
  """
  def select_action(self,q_values,**kwargs):
    return np.argmax(q_values.flatten())

class GreedyEpsilonPolicy(Policy):
  """
  Selects grredy action or with some probablity a random action 
  Parameters
  -----------
  epsilon:float
      Initial probablity of choosing a random action.Updates over time
  """    
  def __init__(self,epsilon):
    assert epsilon <= 1.0
    self.epsilon = epsilon

  def select_action(self,q_values,**kwargs):
    """
    parameters
    -------------
    q_values : array type
       Array of floats representing the q-vlaues for each action
    Returns
    -----------
     type: int   - the action index chosen
    """  
    val = np.random.uniform()
    if val < self.epsilon:
      action = np.random.randint(0, q_values.size)
    else:
      action = np.argmax(q_values.flatten())
    return action

class LinearDecayGreedyEpsilonPolicy(Policy):
  """
  Linearly decaying policy with a paramter. the epsilon decays from a satrt value ot an end value over k steps
  Parameters
  -----------
  start_value:int,float
      The initila value of the parameter
  end_value:int,float
      The value of policy at the nd o fthe decay 
  num_steps: int
      The number of steps over which the policy wil decay         
  """
  def __init__(self,start_value, end_value,num_steps):
    self.cur_step=0
    self.start_value=start_value
    self.end_value = end_value
    self.num_steps = num_steps

  def select_action(self,**kwargs):
    is_training = kwargs['is_training']
    q_values = kwargs['q_values']
    if is_training:
      self.cur_step +=1
    if self.cur_step > self.num_steps:
      cur_epsilon = self.end_value
    else:
      cur_epsilon =self.start_value + (self.cur_step * 1.0/self.num_steps)* (self.end_value - self.start_value)

    if np.random.uniform() < cur_epsilon:
      return np.random.randint(0, q_values.size)
    else:
      return np.argmax(q_vlaues.flattten())

  def reset(self):
    self.cur_step =0
