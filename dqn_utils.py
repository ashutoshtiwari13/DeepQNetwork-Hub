# -*- coding: utf-8 -*-

import preprocessor
import policy
from keras.models import load_model

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

import os
import sys
import copy
import time

NO_OP_STEPS = 30
NO_OF_ITER =100
ITER_EVAL =5e4
MAX_EPISODE_LENGTH = 1e4
NUM_EPISODES =20
ITER_SAVE =5e4
CASE_NORMAL =0
CASE_DOUBLE =1

"""DQN Agent"""
class DQNAgent:
  """
  Class to implement DQN 
  Parameters
  ------------
  q_network :keras.models.Model
      The Q-network Model
  preprocessor : Preprocessor to use, stacked preprocessor can be used implemente din Baseline 

  memory : The replay memory
  gamma : flaot Discount factor - closer to 1 means not taking into consideration 

  target_update_freq : float
    Frequency to update the target network

  num_burn_in : int 
    Number of samples to fill up the replay memory initially

  train_freq :int
    How often to update the Q-ntework 
  batch_size: int 
    How many samples in each minibatch 

  """
  def __init__(self,
               q_network,
               target_q_network,
               preprocessor,
               memory,
               policy,
               gamma,
               target_update_freq,
               num_burn_in,
               train_freq,
               batch_size,
               optimizer,
               loss_func,
               summary_writer,
               checkpoint_dir,
               experiment_id,
               env_name,
               learning_type= CASE_NORMAL):
    self.q_network = q_network
    self.target_q_network = target_q_network
    self.target_q_network.set_weights(self.q_network.get_weights())
    self.compile(optimizer, loss_func)
    self.preprocessor = preprocessor
    self.memory = memory
    self.policy = policy
    self.gamma = gamma
    self.target_update_freq = target_update_freq
    self.num_burn_in = num_burn_in
    self.train_freq = train_freq
    self.batch_size = batch_size
    self.summary_writer = summary_writer
    self.checkpoint_dir = checkpoint_dir
    self.experiment_id = experiment_id
    self.env_name = env_name
    self.training_reward_seen = 0
    
    self.input_batch = np.zeros([batch_size,] + \
                                    list(q_network.input_shape[-3:]), dtype='float')
    self.nextstate_batch = np.zeros([batch_size,] + \
                                        list(q_network.input_shape[-3:]), dtype='float')
    self.learning_type = learning_type 

  def compile(self, optimizer, loss_func):
    self.q_network.compile(optimizer,loss_func)
    self.target_q_network.compile(optimizer,loss_func)

  def q_values(self,state, preproc=None ,network= None):
    """Given a state(or a batch of states)calculate the Q-values. Returns Q-values of the state""" 
    if preproc is None:
      preproc = self.preprocessor
    if network is None:
      network= self.q_network
    return self.q_network.predict(np.expand_dims(preproc.network_process_state(state),0))


  def update_policy(self, itr):
    """Update Policy
    Sample a minibatch , calculate the target values, update your network, and then update your target values.
    """     
    samples = self.memory.sample(self.batch_size)
    num_samples =len(samples)
    assert(num_samples == self.batch_size)

    if self.learning_type == CASE_DOUBLE:
      if np.random.uniform() < 0.5:
        temp = self.q_network
        self.q_network = self.target_q_network
        self.target_q_network = temp

    for i in range(num_samples):
      state, _, _, nextstate,_ = samples[i]
      self.input_batch[i,...] =state
      self.nextstate_batch[i,...] = nextstate
    self.input_batch = self.preprocessor.network_process_state(self.input_batch)
    self.nextstate_batch = self.preprocessor.network_process_state(self.nextstate_batch)
      
      
    target_batch = self.q_network.predict(self.input_batch)
    nextstate_q_values = self.target_q_network.predict(self.nextstate_batch)
   
    if self.learning_type == CASE_DOUBLE:
      nextstate_q_values_live_network = self.q_network.predict(self.nextstate_batch)

    for i in range(num_samples):
      _,action,reward,_,is_terminal = samples[i]
      if is_terminal:
        target_batch[i,action] = reward

      else:
        if self.learning_type == CASE_DOUBLE:
          selected_action = np.argmax(nextstate_q_values_live_network[i].flatten())
          target_batch[i,action]= reward + self.gamma * nextstate_q_values[i,selected_action]
        else:
          target_batch[i,action] = reward + self.gamma * np.max(nextstate_q_values[i])
    
    self.training_reward_seen += sum([i[2] for i in samples])
    if iter % NO_OF_ITER:
      summary_list =[]
      for in range(3):
        s = BytesIO()
        plt.imsave(s, np.mean(self.input_batch[k],axis=-1),format='png')
        img_sum = tf.Summary.Image(
            encoded_image_string=s.getvalue(),
            height=self.input_batch[k].shape[0],
            width=self.input_batch[k].shape[1]
            )
        img.summaries.append(tf.Summary.Value(
            tag ='input/{}'.format(k),image = img_sum))
      self.summary_writer.add_summary(tf.Summary(value= summary_list,global_step =itr))  

    #calculate loss    
    loss = self.q_network.train_on_batch(self.input_batch,target_batch)
    return loss   
  
  def fit(self,env,num_iterations,max_episode_length=None):
    #fill the replaymemory
    self.preprocessor.reset()
    env_current_state =env.reset()
    # env_current_state = self.run_no_op_steps(env)
    env_current_state = self.preprocessor.memory_process_state(env_current_state)

    env = copy.deepcopy(env)
    value_fn = np.random.random((env.action_space.n,))

    for _ in range(self.num_burn_in):
      env_current_state = self.push_replay_memory(
          env_current_state,env,
          policy.UniformRandomPolicy(env.action_space.n),
          is_training=False, value_fn= value_fn)
      
    start_time =time.time()
    for itr in range(num_iterations):
      value_fn = self.q_values(env_current_state,network = self.q_network)
      env_current_state = self.push_replay_memory(
          env_current_state,env,
          self.policy,
          is_training=True, value_fn= value_fn)
      if itr % self.target_update_freq ==0:
        self.target_q_network.set_weights(self.q_network.get_weights())

      if itr % self.train_freq == 0:
        loss = self.update_policy(itr)

      if itr % NO_OF_ITER ==0:
        print('Iteration {:}: Loss {:.12f} ({:.4f} it/sec)'
              '(reward seen : {}'.format(itr,
                                         loss,
                                         NO_OF_ITER * 1.0/(time.time() -start_time),
                                         self.training_reward_seen)      
        start_time = time.time()
        self.summary_writer.add_summary(tf.Summary(value =[
                                                           tf.Summary.Value(
                                                               tag = 'loss',
                                                               simple_value= loss.item())]),
                                        global_step= itr)
        if itr % ITER_EVAL ==0:
          self.evaluate(env,NUM_EPISODES,itr,max_episode_length = MAX_EPISODE_LENGTH)

        if itr % ITER_SAVE ==0:
          self.save(itr)


  def push_replay_memory(self, env_current_state,env,policy,is_training, value_fn):
    dont_reset = False
    env_current_lives = -1
    try :
      env_current_lives = env.env.ale.lives()
    except:
      pass
    action = policy.select_action(q_values= value_fn, is_training= is_training)
    nextstate,reward,is_terminal,debug_info = env.step(action)
    if 'ale.lives' in debug_info:
      if debug_info['ale.lives'] < env_current_lives:
        if not is_terminal:
          dont_reset = True
        is_terminal =True

    reward = self.preprocessor.process_reward(reward)
    nextstate = self.preprocessor.memory_process_state(nextstate)

    self.memory.append(env_current_state,action, reward,nextstate,is_terminal)

    if is_terminal and not dont_reset:
      self.preprocessor.reset()
      nextstate = env.reset()
      nextstate = self.preprocessor.memory_process_state(nextstate)
    return nextstate
    

  def save(slef,itr):
    filename = "%s/%s_run%d_iter%d.h5" %  (self.checkpoint_dir,self.env_name.
                                           self.experiment_id
                                           itr)
    self.q_network.save(filename)

  def load(self,filename):
    self.q_network.save(filename)

  def evaluate(self,env, num_episodes, itr , max_episode_length=None,
               render = False. render_path='', final_eval= False):
    print("Running the evaluation")
    preproc = preprocessor.SequentialPreprocessor(
        [preprocessor.AtariPreprocessor(
            self.q_network.input_shape[1],
            self.preprocessor.preprocessor[0].input_image),
         preprocessor.PreservePreprocessor(self.q_network.input_shape[-1])])

    pol = policy.GreedyEpsilonPolicy(0.05)
    all_stats =[]
    all_rewards = []
    for i in range(num_episodes):
      print('Running episode {}'.format(i))
      if render:
        env = gym.wrappers.Monitor(env,render_path,force= True)

      nextstate = env.reset()
      preproc.reset()
      is_terminal = False 
      stats ={
          'total_reward':0,
          'episode_length':0,
          'max_q_value':0
      } 

      while not is_terminal and stats['episode_length'] < max_episode_length:
        nextstate = preproc.memory_process_state(nextstate)
        q_values = self.q_values(nextstate, preproc)
            action = pol.select_action(q_values=q_values)
            nextstate, reward, is_terminal, _ = env.step(action)
            stats['total_reward'] += reward
            stats['episode_length'] += 1
            stats['max_q_value'] += max(q_values)

        all_stats.append(stats)
        all_rewards.append(stats['total_reward'])
        print('Current mean+ std: {} {}'.format(np.mean(all_rewards),np.std(all_rewards)))


        final_stats = {}
        if render:
          return 
        if final_eval:
          print('Mean reward: {}'.format(np.mean(all_rewards)))
          print('Std reward: {}'.format(np.std(all_rewards)))

          return 

        for key in all_stats[0]:
          final_stats['mean_' + key] = np.mean([i[key] for i in all_stats]).items()  
          self.summary_writer.add_summary(tf.Summary(value=[
                                                            tf.Summary.Value(tag='eval/{}'.format(key),
                                                                             simple_value= final_stats['mean_' + key])]),
                                          global_step=itr)
                                                            
        print('Evaluation result: {}'.format(final_stats)


  def run_no_op_steps(self,env):
    for _ in range(NO_OP_STEPS-1):
        _, _, is_terminal, _ = env.step(0)
        if is_terminal:
          env.reset()
      nextstate, _, is_terminal, _ = env.step(0)
      if is_terminal:
        nextstate = env.reset()
      return nextstate
