'''
Created on Mar 1, 2017

@author: joro
'''
# observation models
import unittest
import numpy as np

from madmom.features.beats_hmm import BarStateSpace, BarTransitionModel,\
    MultiPatternStateSpace, GMMPatternTrackingObservationModel
from madmom.features.bar_notes_hmm import NoteStateSpace, NoteTransitionModel,\
    BarNoteTransitionModel
from madmom.features.bar_note_observation import GMMNotePatternTrackingObservationModel,\
    GMMNoteObservationModel
import pickle
from madmom.models import PATTERNS_BALLROOM
from madmom.features.bar_notes import notestates_to_onsetframes

STEPS_PER_SEMITONE = 2
NUM_SEMITONES = 1
STATES_PER_STEP = 1

class TestGMMNotePatternTrackingObservationModelClass(unittest.TestCase):
    '''
        Test if the logic of combining the existing  PatternObservation with NoteObservation model is correct
        
        When number of notes is 1 and the NoteObsertavion has for all states observation probability=1,
        the joint NotePatternTrackingObservationModel should be equivalent to  BarObservation model
    '''

    def setUp(self):
        bss = BarStateSpace(1, 1, 4) # 1 beat; intervals = [1,4] 
         
        note_state_space = NoteStateSpace(NUM_SEMITONES, STEPS_PER_SEMITONE, STATES_PER_STEP)
        mpss = MultiPatternStateSpace([bss])
        
        ############# set up gmms
        pattern_files = PATTERNS_BALLROOM
        gmms = []
        with open(pattern_files[0], 'rb') as f:
            # Python 2 and 3 behave differently
            # TODO: use some other format to save the GMMs (.npz, .hdf5)
            try:
                # Python 3
                pattern = pickle.load(f, encoding='latin1')
            except TypeError:
                # Python 2 doesn't have/need the encoding
                pattern = pickle.load(f)
        # get the fitted GMMs and number of beats
        gmms.append(pattern['gmms'])
        self.bar_om = GMMPatternTrackingObservationModel(gmms, mpss)
            
        self.note_om = GMMNoteObservationModel(note_state_space)
        
        self.bar_note_om = GMMNotePatternTrackingObservationModel(self.bar_om , self.note_om)
        # two dummy spectral flux observations
        self.obs = np.asarray([[  0.46020326,   4.08274412, 0, 0],[  5.48006916,  22.58586502, 0, 0]], dtype=np.float32)
         

    def test_types(self):
        self.assertIsInstance(self.bar_note_om.pointers, np.ndarray)
        self.assertIsInstance(self.bar_note_om.densities(self.obs), np.ndarray)
        self.assertIsInstance(self.bar_note_om.log_densities(self.obs), np.ndarray)
        self.assertTrue(self.bar_note_om.pointers.dtype == np.uint32)
        self.assertTrue(self.bar_note_om.densities(self.obs).dtype == np.float)
        self.assertTrue(self.bar_note_om.log_densities(self.obs).dtype == np.float)

    def test_compatible(self):

        #         bar_note_om.pointers should be equivalent to bar_om.pointers
        num_note_states = self.note_om.state_space.num_states
        if num_note_states == 1:
            self.assertTrue(np.allclose(self.bar_note_om.pointers,
                                    self.bar_om.pointers))
            # only if note obs. probabilitis. are set to zero
            self.assertTrue(np.allclose(self.bar_note_om.densities(self.obs),
                                    self.bar_om.densities(self.obs)))
        
        else: # test for 2 or more states
            num_gmms = self.bar_om.num_gmms
            
            expected_bar_om_pointers = np.array([])
            for n in range(num_note_states):
                expected_bar_om_pointers = np.hstack((expected_bar_om_pointers, self.bar_om.pointers + n * num_gmms)) 
            
            self.assertTrue(np.allclose(self.bar_note_om.pointers,
                                    expected_bar_om_pointers))
        
        
#         self.assertTrue(np.allclose(self.bar_note_om.log_densities(self.obs),
#                                     [[-np.inf, 0], [-1.20397281, -2.30258508],
#                                      [-1.10866262, -4.60517021],
#                                      [-1.09861229, -np.inf]]))

class TestPostprocessorPathClass(unittest.TestCase):
    
    def test_values(self):
        
        path_indices_note =  [ 281, 279,  280, 280, 280, 281, 279, 280 ]
        f0_values = [35, 48, 48 , 48 , 48 , 0, 0 ,0 ]
        
        onsetFrames = notestates_to_onsetframes(path_indices_note, f0_values)
        self.assertEqual(onsetFrames, [1])