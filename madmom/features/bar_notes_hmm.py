# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

'''
Created on Feb 18, 2017

@author: joro
'''
import numpy as np

from madmom.ml.hmm import TransitionModel


class SingleNoteStateSpace(object):
    def __init__(self):
        self.num_states = 1
        
class NoteStateSpace(object):
    def __init__(self, num_notes):
        self.num_states = 0 # initialize
        
        self.num_notes = num_notes
        for i in range(self.num_notes):
            snss = SingleNoteStateSpace()
            self.num_states += snss.num_states 
            
      
class NoteTransitionModel(object):
    def __init__(self, note_state_space):
        self.state_space = note_state_space
        
        # define some dummy transitions
        if self.num_states == 2:
            self.prev_states = np.array([1,0])
            self.to_states = np.array([0,1])
            self.probs = np.array([1, 1])  
        
        elif self.num_states == 1:
            self.prev_states = np.array([0])
            self.to_states = np.array([0])
            self.probs = np.array([1])
        elif self.num_states == 5:
            self.prev_states = np.array([0,1,2,3,4])
            self.to_states = np.array([1,2,3,4,0])
            self.probs = np.array([1, 1, 1, 1, 1])  
    
    @property
    def num_states(self):
        return self.state_space.num_states        

class BarNoteStateSpace():
    '''
    not used
    '''
    def __init__(self, bar_state_space, note_state_space):
        pass


class BarNoteTransitionModel(TransitionModel):
    '''
    combined transition model: 
    makes a cartesian product of dense bar transition probabilities and dense note transition probabilities 
    the new joint transition model uses indices: 0 to B-1, n=0;  B to 2B-1, n=1; ... n*B to (n+1)*B-1, n 
    where n is counter of note state
    '''



    def __init__(self, bar_transition_model, note_transition_model ):
        '''
        N: num_note_states
        B: num_bar_states
        '''
        
        num_bar_states = bar_transition_model.state_space.num_states
        num_note_states = note_transition_model.state_space.num_states
        
        ##### to_states
        to_bar_states = bar_transition_model.to_states
        to_note_states = note_transition_model.to_states
        self.to_joint_idx = substates_to_flatidx( to_bar_states, to_note_states, num_bar_states, num_note_states)
        
        ##### from_states
        prev_bar_states = bar_transition_model.prev_states
        prev_note_states = note_transition_model.prev_states
        
        self.prev_joint_idx = substates_to_flatidx( prev_bar_states, prev_note_states, num_bar_states, num_note_states)


        ###### probs
        bar_probs = bar_transition_model.probs
        note_probs = note_transition_model.probs
        # cartesian multipl. of probabilities (e.g. matrix), shape : |bar_probs| x |note_probs| 
        mat_prob = bar_probs.reshape(len(bar_probs),1) * note_probs.reshape(1,len(note_probs))
        # matrix to flattened array 
        probs_flattened = mat_prob.flatten(order='F') # order F guarantees that the generated indices increase monotonously, e.g. loop first in bar state space, then (n+1) * bar_state_space    
        self.probs = normalize(self.prev_joint_idx, probs_flattened)
        
        # make the transitions sparse
        transitions = self.make_sparse(self.to_joint_idx, self.prev_joint_idx, self.probs)
        # instantiate a TransitionModel
        super(BarNoteTransitionModel, self).__init__(*transitions)


def substates_to_flatidx( states_1, states_2, max_states_1, max_states_2):
    '''
    convert states_1 (first dimension) and states_2 (second) to flat indices
    Let S (max_states_1) is max val of states_1, U (max_states_2) is max val of states_2 
    Then the new joint indices are: 0 to S-1, u=0;  S to 2*S-1, u=1; ... n*S to (n+1)*S-1, u=(the rest) 
    
    reproduces matlab's sub2ind
    
    Parameters
    --------------------
    states_1: nd.array(N,dtype=np.uint32)
        indices inrange [0:N-1]
    states_2: nd.array(B,)
        indices in range [0:B-1]
    max_states_1 : (<= N)
    max_states_2: (<= B)
    
    Returns
    ---------------------
    idx_flattened: nd.array shape: (max_states_1 x max_states_2 , )
    '''
   
    combined_stateidx_matrix = [np.tile(states_1, len(states_2)), np.repeat(states_2, len(states_1))]
    # cartesian matrix to one-dimensional index in combined state space
    idx_flattened = np.ravel_multi_index(combined_stateidx_matrix, (max_states_1, max_states_2), order='F') # order F guarantees that the generated indices increase monotonously             
    idx_flattened = idx_flattened.astype(np.uint32)
    
    return  idx_flattened

    
def normalize(from_states, probs):
    '''
    the probabilities transition probs originating from same state in array from_states have to sum to 1 
    NOTE: supports only non-zero states
    '''
    index_sets = [np.argwhere(i==from_states) for i in np.unique(from_states)]
    for index_set in index_sets: # set of unique from_state indices
        probs[index_set] /= np.sum(probs[index_set])
    return probs
    
    

