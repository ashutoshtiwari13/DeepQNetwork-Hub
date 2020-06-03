# -*- coding: utf-8 -*-

#Google Colab File 
#created by ashutoshtiwari13

class Sample:
  """
  The MDP of the standard form (s, a , r , s', terminal_bool)
  Params :
  --------
  state (s) : Represents the state of the MDP before taking an action. 
              Type - Numpy array 
  action (a) : Represents the action taken by the agent.Can be integer (for disreate action space
                ) , float (for continous action space), Tuple (for a parmeterized action MDP this wil be a tuple 
              contiaining the action and its associated paramters)
  reward(r) : The reward recieved as a result of the action taken by the agent in the given state transiontioning into the resultant state
              Type - float
  next_state (s'): This the state the agent transitions to after executing the 'action' in 'state'.

  terminal_bool : True if this action finished the episode (if episodic).False otherwise.

  """
  pass

class Preprocessor:
  """
  Interface for defining the preprocessing steps.

  Can be used to perform a fixed operation on the raw state from an environment.
  Eg - grayscale conversion or downsampling of image fed to a Convnet based NNs

  It is important to implement preprocessor as a class in order to let them have their own internal states.
  Eg - Used for things like AtariPreproccessor which maximize over k frames

  Thess internal states when used for keeping a sequence of inputs like Atari, could have a reset() call when a 
  new episode begins so that the state doesn't leak in form episode to episode.
 
  """

  def network_process_state(self, state):
    """
    Preprocess the state before feedng into the network.
    Should be called just before the action is selected

    Parameters :
    -----------------
    state : np.ndarray 
        A single state from the environment.

    returns :
    -------------------
    processed_state: np.ndarray
    """
    return state


  def memory_process_state(self, state):
    """
    Preprocess the state before giving it to th ereplay memory.
    Should be called just before the action is selected

    This is a different method from the process_state_for_network
    because the replay memory may require a different storage
    format to reduce memory usage. For example, storing images as
    uint8 in memory and the network expecting images in floating
    point.

    Parameters :
    -----------------
    state : np.ndarray 
        A single state from the environment.

    returns :
    -------------------
    processed_state: np.ndarray
    """
    return state

  def process_batch(self, samples):
    """
    If the replay memory storage format is differnet from the inputs to your network
   ,which is mostly the case , this function can be applied to the sampled batch before
    running it thorugh the update function.

    Parameters :
    -----------------
    samples: list()
        List of samples to process

    returns :
    -------------------
    processed_samples: list()
    """
    return samples 

  def process_reward(self, reward):
    """
    Process the reward obtained on state transition .
    Eg -Reward clipping i.e. instead of taking real score , take the sign of the delta of the score
  
    Parameters :
    -----------------
    reward : float
        Reward to proces

    returns :
    -------------------
    processed_reward: float
    """
    return reward  

  def reset(self):
    """
    Reset any internal state 
    Should be called at the start of any new episode. Helps to take history snapshots.
    """
    pass

class ReplayMemory:
  """
  Used as a interface for the replay memory.
  Generally, the replay memory has implemented the __iter__, __len__, __getitem__ methods.

  If you are storing raw BaselineSample objects in the memory , then the end_episode method is not needed.
  tweaking the append method helps in such a case. We can just randomly draw samples saved in the memory)

  However , the above approach will waste a lot of memory( as states will be stored multiple times in s as next state and
  then s' as state, etc.) . Depending upon the machine resources in use we will want to store BaselineSamples in a memory efficient way.

  Methods :
  ----------------------
  append( state , action ,reward, debug_info=None):
     Add a Sample to the replay memory.

  end_episode(final_state, terminal_bool, debug_info=None):
     Set the terminal state of an episode and mark whether it was a true terminal state(i.e env returned terminal_bool=True),
     of it is an artificial terminal state (i.e agent quit the episode early, but gaent could have kept running episdode)

  sample(batch_size,indexes=None):
    Return list of samples from the memory .Each class will implement a different method of choosing the samples.
    We can specify the samples indexes manually too!

   clear()
      reset the memory .Deletes all references to the samples.       
  """
  def __init__(self, max_size, window_length):
    """
    """
    pass

  def append(self, state,action,reward):
    pass

  def end_episode(self, final_state,terminal_bool):
    pass

  def sample(self, batch_size, indexes=None):
    raise NotImplementedError('Not implemented , method should be overridden')

  def clear(self):
    pass
