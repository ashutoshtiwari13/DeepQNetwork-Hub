# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

from baseline import Preprocessor

class AtariPreprocessor(Preprocessor):
  """
  Converts images to grayscale and downscales.

  Parameters :
  -------------
  new_size:2 element tuple
      The size that each image in the state should be scaled to. e.g (84,84) will make
      each output image of the same shape (84,84)

  Note :
   Steps from Paper :
   Human level Control through Deep Reinforcement Learning, 2015    
  """

  def __init__(self,new_size,input_image):
    self.new_size=new_size 
    self.resize_shape = (110,84)  
    self.input_image = input_image

  def network_process_state(self,state):
    """
    Convert to grayscale and store as float32. 
    PIL is used for image conversions.
    """
    state = state.astype('float')
    if self.input_image:
      state = state - 128.0
      state = state / 255.0

    return state  

  def memory_process_state(self,state):
    """
    Convert to grayscale and store as uint8. 
    PIL is used for image conversions.
    """

    img = Image.fromarray(state,'RGB')
    img = img.convert('L')         #convert to gray
    img = img.resize((self.new_size,self.new_size),Image.ANTIALIAS)
    """
    In order to resize by left , right , bottom , top
    img =img.resize(self.resize_shape)
    width , height = img.size
    new_width= self.new_size
    new_length= self.new_size

    left= max((width - new_width)/2, 0)
    right= max((width - new_width)/2, width)
    bottom= max((height - new_height)/2, height)
    top= max((height - new_height)/2, 0)

    img= img.crop((left,top,right,bottom))
    """
    img = np.array(img).astype('unit8')
    return img

  def process_batch(self, samples):
    """
    The batches from the replay memory are unit8 , convert to flaot32
    same as network_process_state but works on a batch of samples from the replay memory.
    We need to convert both the state and next state values
    """

    return [self.network_process_state(sample) for sample in samples ]

  def process_reward(self, reward):
    """Clipping reward between -1 and 1"""
    return np.sign(reward)        #sign of the delta of rewards is taken

class PreservePreprocessor(Preprocessors):
  """
  Keep the last k states.

  Parameters
  ------------------
  preserve_length:int
    Number of previous states to prepend to state being processed.
  """
  def __init__(self, preserve_length=1):
    self.preserve = np.zeros((84,84,preserve_length))
    self.preserve_length= preserve_length

  def memory_process_state(self,state):
    self.preserve[...,0]=state
    self.preserve = np.roll(self.preserve,-1,axis=-1)
    return self.preserve.copy()

  def reset(self):
    """
    can be used to reset the preserve sequence on the start of 
    the new episode
    """
    self.preserve = np.zeros((84,84,preserve_length))

class SequentialPreprocessor(Preprocessor):
  """
  The preprocessors can be stacked to be called in succession by using a class.
  Eg. Calling the network_process_state and a sequential call to AtariPre and PreservePre

  This class will help in sequential call as follows,
  state= atari.network_process_state(state)
  return preserve.network_process_state(state)
  """
  def __init__(self, preprocessors):
    self.preprocessors = preprocessors

  def network_process_state(self,state):
    """
    Run the pre-processor sequentially
    """  
     for preproc in self.preprocessors:
       state = preproc.network_process_state(state)
       return state

  def memory_process_state(self,state):
    for preproc in self.preprocessors:
       state = preproc.memory_process_state(state)
       return state

  def reset(self):
    for preproc in self.preprocessors:
      preproc.reset()
