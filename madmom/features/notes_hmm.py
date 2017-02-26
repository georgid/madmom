'''
Created on Feb 18, 2017

@author: joro
'''
import numpy as np

from madmom.ml.hmm import TransitionModel, ObservationModel


class SingleNoteStateSpace(object):
    def __init__(self):
        self.num_states = 3
        
class NoteStateSpace(object):
    def __init__(self):
        self.num_states = 0
        self.states = [] 
        
        self.num_notes = 1
        for i in range(self.num_notes):
            snss = SingleNoteStateSpace()
            self.num_states += snss.num_states 
            
      
class NoteTransitionModel(object):
    def __init__(self, note_state_space):
        self.state_space = note_state_space
        
        # define some dummy transitions
        self.states = np.array([0,1,1])
        self.prev_states = np.array([0,0,1])
        self.probabilities = np.array([0.1, 0.9, 1])  
        
        self.states = [0,1,1]
        self.prev_states = [0,0,1]
        self.probabilities = [0.1, 0.9, 1]
        

class BarNoteStateSpace():
    def __init__(self):
        pass

class BarNoteTransitionModel(TransitionModel):
    '''
    combined transition model: 
    makes a cartesian product of dense bar transition probabilities and dense note transition probabilities 
    the new unified transition model uses indices: 1 to B, n=0;  B+1 to 2B, n=1, (n-1)B +1 to nB, n 
    where n is count of note state
    '''



    def __init__(self, bar_transition_model, note_transition_model ):
        '''
        N: num_note_states
        B: num_bar_states
        '''
        
        num_bar_states = bar_transition_model.state_space.num_states
        num_note_states = note_transition_model.state_space.num_states
        
        ##### to_states
        to_bar_states = bar_transition_model.states
        to_note_states = note_transition_model.states
        to_idx_flattened = substates_to_flatidx( to_bar_states, to_note_states, num_bar_states, num_note_states)
        
        ##### from_states
        prev_bar_states = bar_transition_model.prev_states
        prev_note_states = note_transition_model.prev_states
        prev_idx_flattened = substates_to_flatidx( prev_bar_states, prev_note_states, num_bar_states, num_note_states)


        ###### probs
        bar_probs = bar_transition_model.probabilities
        note_probs = note_transition_model.probabilities
        # cartesian multipl. of probabilities (e.g. matrix), shape : |N| x |B|
        mat_prob = note_probs.reshape(len(note_probs),1) * bar_probs.reshape(1,len(bar_probs))
        # matrix to flattened array 
        probs_flattened = mat_prob.flatten()
        probs_flattened = normalize(prev_idx_flattened, probs_flattened)
        
        # make the transitions sparse
        transitions = self.make_sparse(to_idx_flattened, prev_idx_flattened, probs_flattened)
        # instantiate a TransitionModel
        super(BarNoteTransitionModel, self).__init__(*transitions)


class BarNoteTransitionModel_old(TransitionModel):
    '''
    extends the bar transition model: multiplies each bar state by the L states
    fist bar point state is in (0,L-1), second in (1*L, 2L-1) , n-th in ( (n-1)*L, n*L-1)
    '''
    def __init__(self, bar_transition_model, note_transition_model ):
        
        self.bar_transition_model = bar_transition_model
        #  note bar_states 
        self.note_state_space = note_transition_model.state_space
        num_note_transitions  = len(note_transition_model.states) # non-zero prob. transitions
        
#         bar_states = np.repeat(self.bar_transition_model.bar_states, num_note_transitions, axis=1   )
        bar_states = self.bar_transition_model.states
        
#         prev_bar_states = np.repeat(self.bar_transition_model.prev_bar_states * num_note_transitions)
        prev_bar_states = self.bar_transition_model.prev_states
        
        bar_probabilities = self.bar_transition_model.probabilities
        
        # dense transitions of combined bar space and note space  
        bar_note_states = []
        prev_bar_note_states = []
        bar_note_probs = []
        
        ### convert to new state numbers: each bar state spans new num_note_states  
        bar_states = bar_states * self.note_state_space.num_states
        prev_bar_states = prev_bar_states * self.note_state_space.num_states
        for state, prev_state, prob in zip(bar_states, prev_bar_states, bar_probabilities):
            for l in range(num_note_transitions):
                bar_note_states.append(state + note_transition_model.states[l])
                prev_bar_note_states.append(prev_state + note_transition_model.prev_states[l])
                bar_note_probs.append(prob * note_transition_model.probabilities[l])
        
                
        bar_note_states = np.array(bar_note_states)
        prev_bar_note_states = np.array(prev_bar_note_states)
        bar_note_probs = np.array(bar_note_probs)

        bar_note_probs = normalize(prev_bar_note_states, bar_note_probs)
        
        # make the transitions sparse
        transitions = self.make_sparse(bar_note_states, prev_bar_note_states, bar_note_probs)
        # instantiate a TransitionModel
        super(BarNoteTransitionModel_old, self).__init__(*transitions)


def substates_to_flatidx( states_1, states_2, num_states_1, num_states_2):
    '''
    convert states_1 (first dimension) and states_2 (second) to flat indices
    '''
    # cartesian product (e.g. matrix) of two from_states, shape : |N| x |B|
    combined_stateidx_matrix = [np.tile(states_1, len(states_2)), np.repeat(states_2, len(states_1))]
    # cartesian matrix to one-dimensional index in combined state space
    idx_flattened = np.ravel_multi_index(combined_stateidx_matrix, (num_states_1, num_states_2))             
    
    return  idx_flattened

    
def normalize(from_states, probs):
    '''
    the probabilities transition probs originating from same state in array from_states have to sum to 1 
    '''
    index_sets = [np.argwhere(i==from_states) for i in np.unique(from_states)]
    for index_set in index_sets: # set of unique from_state indices
        probs[index_set] /= np.sum(probs[index_set])
    return probs
    
    

class GMMNoteTrackingObservationModel(ObservationModel):
    """
    Observation model for GMM based beat tracking with a HMM.

    Parameters
    ----------
    pattern_files : list
        List with files representing the rhythmic patterns, one entry per
        pattern; each pattern being a list with fitted GMMs.
    state_space : class:`NoteStateSpace` instance
         All-notes state space.

    References
    ----------


    """

    def __init__(self, pattern_files, state_space):
        # save the parameters
        self.pattern_files = pattern_files
        self.state_space = state_space
        # define the pointers of the log densities
#         pointers = np.zeros(state_space.num_states, dtype=np.uint32)
      

#         gmms = pattern_files[0]
            # number of fitted GMMs for this pattern
        pointers = np.array(range(state_space.num_states))

        # instantiate a ObservationModel with the pointers
        super(GMMNoteTrackingObservationModel, self).__init__(pointers)

    def log_densities(self, observations):
        """
        Computes the log densities of the observations using (a) GMM(s).

        Parameters
        ----------
        observations : numpy array
            Observations (i.e. multi-band spectral flux features).

        Returns
        -------
        numpy array
            Log densities of the observations.

        """
        # number of GMMs of all patterns
        num_gmms = sum([len(pattern) for pattern in self.pattern_files])
        # init the densities
        log_densities = np.empty((len(observations), num_gmms), dtype=np.float)
        # define the observation densities
        i = 0
        for pattern in self.pattern_files:
            for gmm in pattern:
                # get the predictions of each GMM for the observations
                log_densities[:, i] = gmm.score(observations)
                i += 1
        # return the densities
        return log_densities

