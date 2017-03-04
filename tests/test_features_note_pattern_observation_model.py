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
class TestGMMNotePatternTrackingObservationModelClass(unittest.TestCase):
    '''
        Test if the logic of combining the existing  PatternObservation with NoteObservation model is correct
        
        When number of notes is 1 and the NoteObsertavion has for all states observation probability=1,
        the joint NotePatternTrackingObservationModel should be equivalent to  BarObservation model
    '''

    def setUp(self):
        bss = BarStateSpace(1, 1, 4) # 1 beat; intervals = [1,4] 
        self.NUM_NOTE_STATES = 2 
        note_state_space = NoteStateSpace(self.NUM_NOTE_STATES)
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
            
        note_om = GMMNoteObservationModel(note_state_space)
        
        self.bar_note_om = GMMNotePatternTrackingObservationModel(self.bar_om , note_om)
        # two dummy spectral flux observations
        self.obs = np.asarray([[  0.46020326,   4.08274412],[  5.48006916,  22.58586502]], dtype=np.float32)
         

    def test_types(self):
        self.assertIsInstance(self.bar_note_om.pointers, np.ndarray)
        self.assertIsInstance(self.bar_note_om.densities(self.obs), np.ndarray)
        self.assertIsInstance(self.bar_note_om.log_densities(self.obs), np.ndarray)
        self.assertTrue(self.bar_note_om.pointers.dtype == np.uint32)
        self.assertTrue(self.bar_note_om.densities(self.obs).dtype == np.float)
        self.assertTrue(self.bar_note_om.log_densities(self.obs).dtype == np.float)

    def test_compatible(self):

        #         bar_note_om.pointers should be equivalent to bar_om.pointers
        if self.NUM_NOTE_STATES == 1:
            self.assertTrue(np.allclose(self.bar_note_om.pointers,
                                    self.bar_om.pointers))
            # only if same probs.
            self.assertTrue(np.allclose(self.bar_note_om.densities(self.obs),
                                    self.bar_om.densities(self.obs)))
        
        elif self.NUM_NOTE_STATES == 2: # TODO test for 2>1: do NUM_NOTE_STATES times hstack
            num_gmms = self.bar_om.num_gmms
            expected_bar_om_pointers = np.hstack((self.bar_om.pointers, self.bar_om.pointers + num_gmms)) 
            self.assertTrue(np.allclose(self.bar_note_om.pointers,
                                    expected_bar_om_pointers))
        
        
#         self.assertTrue(np.allclose(self.bar_note_om.log_densities(self.obs),
#                                     [[-np.inf, 0], [-1.20397281, -2.30258508],
#                                      [-1.10866262, -4.60517021],
#                                      [-1.09861229, -np.inf]]))

 