# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

import numpy as np
from madmom.ml.hmm import ObservationModel
from madmom.features.bar_notes_hmm import substates_to_flatidx

'''
Created on Feb 28, 2017

@author: joro
'''
class GMMNoteObservationModel(ObservationModel):
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

    def __init__(self,  state_space):
        # save the parameters
        self.state_space = state_space
        # define the pointers of the log densities
#         pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        pointers = np.array( range(state_space.num_states), dtype=np.uint32)
        # each note state points to exactly one GMM
        self.num_gmms = state_space.num_states 
        # instantiate a ObservationModel with the pointers
        super(GMMNoteObservationModel, self).__init__(pointers)

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
        # init the densities
        log_densities = np.zeros((len(observations), self.state_space.num_states), dtype=np.float)
        # return the densities
        return log_densities

class GMMNotePatternTrackingObservationModel(ObservationModel):
    """
    Observation model Combined from two GMM-based models 
    Used for simultaneous beat and note tracking with a HMM.
    p(y^f, y^p | x) = p(y^f | x) * p (y^p | x)

    """

    def __init__(self,  bar_om, note_om ):
        
        # define the pointers of the joint log densities
#         pointers = np.zeros(bar_om.state_space.num_states * note_om.state_space.num_states, dtype=np.uint32)
        
        self.bar_om = bar_om
        self.bar_note_om = note_om
        bar_pointers = bar_om.pointers
        note_pointers = note_om.pointers
        num_note_gmms = note_om.num_gmms
        
        self.patterns_num_gmms = []  
        
        # number of fitted GMMs of bar observation for current pattern 
        num_gmms_bar = len(bar_om.pattern_files[0])
        self.patterns_num_gmms.append(num_gmms_bar)
        
        # pointers to GMMs for current pattern
        bar_pointers_pattern =  bar_pointers[bar_om.state_space.state_patterns == 0]
        ## pointers of joint state space; the states are the indices  
        pointers_flattened = substates_to_flatidx( bar_pointers_pattern, note_pointers, num_gmms_bar, num_note_gmms) 
            
        # instantiate a ObservationModel with the pointers
        super(GMMNotePatternTrackingObservationModel, self).__init__(pointers_flattened)

    def log_densities(self, observations):
        """
        Cartesian product of computed densities with bar observation model and
        densities with note observation model 
        Parameters
        ----------
        observations : numpy array
            Observations (i.e. multi-band spectral flux features).

        Returns
        -------
        numpy array
            combined Log densities of the observations.

        """
        
        # compute densities with the two observation models
        bar_log_densities = self.bar_om.log_densities(observations) # size O x (num_bar_gmms -> for each pattern)
        note_log_densities = self.bar_note_om.log_densities(observations) # size O x num_note_gmms
        

#         num_bar_gmms_prev_pattern = 0
        for p in range(len(self.patterns_num_gmms)):
            curr_num_bar_gmms = self.patterns_num_gmms[p] # for current pattern
            joint_log_densities = self._create_join_log_densities( bar_log_densities, note_log_densities, curr_num_bar_gmms)
        # return the densities
        return joint_log_densities


    def _create_join_log_densities(self, bar_log_densities, note_log_densities, num_bar_gmms):
        '''
        p(y^f, y^p | x_l) = p(y^f | x_b) * p (y^p | x_n), for each b \in |B| and for each n \in |N|
        
        '''
        num_observations = bar_log_densities.shape[0]
        num_note_gmms = self.bar_note_om.num_gmms
        joint_log_densities = np.empty((num_observations, num_bar_gmms * num_note_gmms), dtype=np.float)
    
        indices_flattened = np.arange(joint_log_densities.shape[1])
        (indices_bar, indices_note) = np.unravel_index(indices_flattened, (num_bar_gmms, num_note_gmms), order='F')
                    
        joint_log_densities[:,indices_flattened] = bar_log_densities[:,indices_bar] + note_log_densities[:,indices_note]
        return joint_log_densities