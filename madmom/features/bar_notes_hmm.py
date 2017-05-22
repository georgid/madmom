# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
import os
import sys
import logging

'''
Created on Feb 18, 2017

@author: joro
'''
import numpy as np

from madmom.ml.hmm import TransitionModel

# add path  to pyPYIN for transition model
parentDir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__) ), os.path.pardir, os.path.pardir, os.path.pardir)) 
pathpYIN = os.path.join(parentDir, 'pypYIN')
if pathpYIN not in sys.path:
    sys.path.append(pathpYIN)
from pypYIN.MonoNoteHMM import MonoNoteHMM  


class SingleNoteStateSpace(object):
    def __init__(self, states_per_step):
        self.num_states = states_per_step
        
class NoteStateSpace(object):
    '''
    
    '''
    def __init__(self, num_semitones, steps_per_semitone, states_per_step):
        '''
        num_distinct_notes: num
        '''
        
        self.states_per_step = states_per_step 
        snss = SingleNoteStateSpace(states_per_step)
        self.num_states = num_semitones * steps_per_semitone * states_per_step
        
        self.steps_per_semitone = steps_per_semitone
        self.num_semitones = num_semitones 
      
      
class NoteTransitionModel(object):
    def __init__(self, note_state_space, usul_type, beat_aware=0):
        self.state_space = note_state_space
        self.usul_type = usul_type
        
        if beat_aware:
            self._construct_trans_probs()
        else:
            self.build_dummy_note_transition_probs()

        
  
    
    def _construct_trans_probs(self):
        '''
        construct transProbs matrix with pyPYN code. 
        Then restructure to make a |B| x |N| matrix, e.g. different note transitions for each bar position b. 
        
        beat_trans_probs: shape: (num_usul_beats,)
            transprobs at the B beats for this usul
        '''
            
        with_bar_dependent_probs=1
        hopTime = 1 # dummy
        
        hmm_notes = MonoNoteHMM(  self.state_space.steps_per_semitone, self.state_space.num_semitones,  with_bar_dependent_probs, hopTime, self.usul_type)
        hmm_notes.build_trans_probs(with_bar_dependent_probs)

        self.prev_states = hmm_notes.fromIndex
        self.to_states = hmm_notes.toIndex
        
        self.probs = hmm_notes.transProbs[0,-1] # beat-unaware deafault probs. as in model of Mauch
        self.beat_trans_probs =  hmm_notes.transProbs[:,0] # the trnas probs at the B usul beats. 0 means zero dist from beat  
        
        
    def build_dummy_note_transition_probs(self):
            '''
            simple transitions for just a few note states
            '''
            if self.num_states == 3:
                self.prev_states = np.array([0,1,2])
                self.to_states = np.array([1,2,3])
                self.probs = np.array([1, 1, 1])
            
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



    def __init__(self, bar_transition_model, note_transition_model, dependent_on_bar=1 ):
        '''
        N: num_note_states
        B: num_bar_states
        '''
        
        logging.info('combining bar_trasitions and note transitions in one transition model... ')
        self.bar_transition_model = bar_transition_model
        self.note_transition_model = note_transition_model
        
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
        mat_prob = self.build_trans_probs(dependent_on_bar)
        # matrix to flattened array 
        self.probs = mat_prob.flatten(order='F') # order F guarantees that the generated indices increase monotonously, e.g. loop first in bar state space, then (n+1) * bar_state_space    
#         self.probs = normalize(self.prev_joint_idx, probs_flattened)
        
        # make the transitions sparse
        transitions = self.make_sparse(self.to_joint_idx, self.prev_joint_idx, self.probs)
        # instantiate a TransitionModel
        super(BarNoteTransitionModel, self).__init__(*transitions)
    
    def build_trans_probs(self,  dependent_on_bar):
        '''
        
        Returns
        -----------------
        joint_trans_probs: shape : |dense_bar_probs| x |dense_note_probs|
             cartesian multipl. of dense probabilities for bar and dense probabilities for note transitions 
        
        '''
        bar_probs = self.bar_transition_model.probs
        note_probs = self.note_transition_model.probs
        
        if dependent_on_bar: 
            bar_state_positions = self.bar_transition_model.state_space.state_positions
            bar_prev_states = self.bar_transition_model.prev_states
            
            positions_prev_states = bar_state_positions[bar_prev_states] # bar positions of prev_states 
            indices_beats = np.argwhere( np.equal(np.mod(positions_prev_states, 1), 0) )[:,0] # 
            
            beat_positions = positions_prev_states[indices_beats].astype(int)
            # init by tiling the note_probs |bar_probs| times
            joint_trans_probs = np.tile(note_probs, (len(bar_probs),1))
            
            # replace with bar-dependent probs at beat positions
            beat_trans_probs = self.note_transition_model.beat_trans_probs
            trans_probs = beat_trans_probs[beat_positions]
            for (i,trans_prob) in zip(indices_beats, trans_probs):
                joint_trans_probs[i,:] = trans_prob
            
            joint_trans_probs *=  bar_probs.reshape(len(bar_probs), 1)
            
        else: # 
            joint_trans_probs = bar_probs.reshape(len(bar_probs), 1) * note_probs.reshape(1, len(note_probs))
            
        normalized_joint_trans_probs = normalize(joint_trans_probs, self.bar_transition_model.prev_states, self.note_transition_model.prev_states)
        
        return normalized_joint_trans_probs

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

    
def normalize(joint_trans_probs, bar_from_states, note_from_states):
    
    ## groups of indices of dense from. states for a bar 
    unique_bar_from_states = np.unique(bar_from_states)
    groups_indicies_bar_from_states = [np.argwhere(bar_from_states==i) for i in unique_bar_from_states]
    
    ## groups of indices of dense from states for a note 
    unique_note_from_states = np.unique(note_from_states)
    groups_indices_note_from_states = [np.argwhere(note_from_states==i) for i in unique_note_from_states]
    
    for group_indices_from_bar in groups_indicies_bar_from_states: # combine the groups of unique indices for the two dimensions
        for group_indices_from_note in groups_indices_note_from_states:
            index_array = to_index_array(group_indices_from_bar,group_indices_from_note)
            sum_probs = np.sum( joint_trans_probs[ index_array ] )
            if sum_probs != 1:
                joint_trans_probs[index_array] /= sum_probs
            
    return joint_trans_probs

def to_index_array(group_indices_from_bar,group_indices_from_note):
    '''
    convert the numpy arrays to index arrays
    '''
    return (group_indices_from_bar, group_indices_from_note.T)
    
    
def normalize_(from_states, probs):
    '''
    @deprecated
    the probabilities transition probs originating from same state in array from_states have to sum to 1 
    NOTE: supports only non-zero states
    '''
    unique_from_states = np.unique(from_states)
    for i in unique_from_states:
#         index_sets = np.argwhere(from_states==i)
        print i
    for index_set in index_sets: # set of unique from_state indices
        probs[index_set] /= np.sum(probs[index_set])
    return probs
    
    

