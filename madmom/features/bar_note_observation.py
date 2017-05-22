# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

import numpy as np
from madmom.ml.hmm import ObservationModel
from madmom.features.bar_notes_hmm import substates_to_flatidx
from madmom.processors import SequentialProcessor
import sys
import os
import json


# parentDir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__) ), os.path.pardir,  os.path.pardir,  os.path.pardir))
# path_Alignment_duration =     os.path.join(parentDir, 'AlignmentDuration')
# if path_Alignment_duration not in sys.path:
#         sys.path.append(path_Alignment_duration)
from demo import extract_predominant_vocal_melody 
from pypYIN.MonoNoteHMM import MonoNoteHMM  
import logging

'''
Created on Feb 28, 2017

@author: joro
'''

class CombinedFeatureProcessor(SequentialProcessor):
    '''
    The extracted feature with SequentialProcessor.process() is combined with another externally extracted feature
    
    Returns
    -------------------------
    data_plus_external_feature: shape(data.shape[0], data.shape[1] + EXTERNAL_FEATURE_SHAPE)
    
    Example usage: 
    in_processor = CombinedFeatureProcessor([sig, frames, filt, log, diff, mb])
    '''
    def set_params(self, pitch_file_URI, hopsize, framesize, end_ts):
        self.hopsize_pitch =  hopsize
        self.framesize_pitch = framesize
        self. pitch_file_URI  = pitch_file_URI
        self.end_ts = end_ts
        
    def process(self, data):
        
        data = super(CombinedFeatureProcessor, self).process(data)
        # extract pitch
        MIDI_pitch_contour_and_prob = self._extract_midi_pitch(self.pitch_file_URI, self.hopsize_pitch, self.framesize_pitch, self.end_ts)
        
        if MIDI_pitch_contour_and_prob.shape[0] > data.shape[0]:
            to_frame_idx = data.shape[0]
            logging.warning('MIDI pitch has length {} whihch is longer than spectral flux features'.format(MIDI_pitch_contour_and_prob.shape[0]) )
#             sys.exit( 'Not implemented. midi pitch frames are {} and spectral flux are {}'.format(MIDI_pitch_contour_and_prob.shape[0], data.shape[0]) )
        else: 
            to_frame_idx = MIDI_pitch_contour_and_prob.shape[0]
        # combine
        data_plus_external_feature = np.hstack((data[:to_frame_idx,:], MIDI_pitch_contour_and_prob[:to_frame_idx,:] ))
        return data_plus_external_feature
    
    def _extract_midi_pitch(self, audio_file_URI, hopSize, frameSize, end_ts):
        '''
        extract pitch feature
        '''
        PITCH_PROB = 0.9 # global as in [Mauch] paper
        
        pitch_file_URI = audio_file_URI[:-4] + '.pitch_audio_analysis' 
        if os.path.isfile(pitch_file_URI):
            with open(pitch_file_URI, 'r') as f1:
                estimatedPitch_vocal = json.load(f1)
                estimatedPitch_vocal = np.array(estimatedPitch_vocal)
        else:
            estimatedPitch_vocal = extract_predominant_vocal_melody(audio_file_URI, hopSize, frameSize, None, end_ts)
            # write to file
            with open(pitch_file_URI, 'w') as f:
                json.dump(estimatedPitch_vocal.tolist(), f)
        
        ## convert to MIDI    
        MIDI_pitch_contour_and_prob = np.zeros((len(estimatedPitch_vocal),2)) 
        MIDI_pitch_contour_and_prob[:,0] = estimatedPitch_vocal
        
        for iFrame in range(len(estimatedPitch_vocal)):
            if estimatedPitch_vocal[iFrame] > 0:  # zero or negative value (silence) remains with 0 probability and negative frequency in Herz
                MIDI_pitch_contour_and_prob[iFrame][0] = 12 * np.log(estimatedPitch_vocal[iFrame]/440.0)/np.log(2.0) + 69
                MIDI_pitch_contour_and_prob[iFrame][1] = PITCH_PROB # constant voicing probability = 0.9
        
        return MIDI_pitch_contour_and_prob
    
    
    
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
        
        
        with_bar_dependent_probs = 1 # dummy
        hopTime = 1 # dummy
        usul_type = 'duyek' # dummy
        self.hmm_notes = MonoNoteHMM(self.state_space.steps_per_semitone, self.state_space.num_semitones,  with_bar_dependent_probs, hopTime, usul_type)
        self.hmm_notes.build_obs_model()
        
         
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
        numpy array: nd.array shape()
            Log densities of the observations.

        """
        # dummy densities: TODO define densities here
#         log_densities = np.zeros((len(observations), self.state_space.num_states), dtype=np.float)
        
        densities = self.hmm_notes.calculatedObsProb(observations)
        densities = self.hmm_notes.normalize_obs_probs(densities, observations)
        densities = densities.T
        
        # return the densities
        return np.log(densities)


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
        num_note_gmms = note_om.num_gmms # if you change number of note gmms, you might need to change also logic of substates_to_flatidx
        
        which_pattern = 0 # it works with one pattern only
        # number of fitted GMMs of bar observation for current pattern 
        self.num_gmms_bar = len(bar_om.pattern_files[which_pattern])
        
        # pointers to GMMs for current pattern
        bar_pointers_pattern =  bar_pointers[bar_om.state_space.state_patterns == which_pattern]
        ## pointers of joint state space; the states are the indices  
        pointers_flattened = substates_to_flatidx( bar_pointers_pattern, note_pointers, self.num_gmms_bar, num_note_gmms) 
            
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
        if observations.shape[1] != 4:
            sys.exit('expecting a 4-dimensioanl feature: \
             two-dimnesional spectral flux + one-dimnesional pitch. Got intstead {} dimensional featrure '.format(observations.shape[1]))
        # compute densities with the two observation models
        bar_log_densities = self.bar_om.log_densities(observations[:,:2]) # size O x (num_bar_gmms -> for each pattern)
        note_log_densities = self.bar_note_om.log_densities(observations[:,2:]) # size O x num_note_gmms
        

        curr_num_bar_gmms = self.num_gmms_bar # for current pattern
        joint_log_densities = self._create_join_log_densities( bar_log_densities, note_log_densities, curr_num_bar_gmms)
        # return the densities
        return joint_log_densities


    def _create_join_log_densities(self, bar_log_densities, note_log_densities, num_bar_gmms):
        '''
        joint_log_densities: the LHS of
        p(y^f, y^p | x) = p(y^f | b) * p (y^p | n), for each b \in |num_bar_gmms| and for each n \in |num_note_gmms|
        
        Parameters
        -------------------
        bar_log_densities shape (num_obs, num_bar_states)
        note_log_densities shape (num_obs, num_note_states)
        
        
        '''
        num_observations = bar_log_densities.shape[0]
        num_note_gmms = self.bar_note_om.num_gmms
        joint_log_densities = np.empty((num_observations, num_bar_gmms * num_note_gmms), dtype=np.float)
    
        indices_flattened = np.arange(joint_log_densities.shape[1])
        (indices_bar, indices_note) = np.unravel_index(indices_flattened, (num_bar_gmms, num_note_gmms), order='F')
                    
        joint_log_densities[:,indices_flattened] = bar_log_densities[:,indices_bar] + note_log_densities[:,indices_note]
        return joint_log_densities