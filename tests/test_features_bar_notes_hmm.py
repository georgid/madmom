'''
Created on Mar 1, 2017

@author: joro
'''
import unittest
import numpy as np
from madmom.features.beats_hmm import BarStateSpace, BarTransitionModel
from madmom.ml.hmm import TransitionModel
from madmom.features.bar_notes_hmm import NoteStateSpace, NoteTransitionModel,\
    BarNoteTransitionModel

    
 
class TestNoteTransitionModelClass(unittest.TestCase):
        
        def setUp(self):
            self.NUM_NOTE_STATES = 5
            self.NUM_NOTE_STATES = 2
            self.NUM_NOTE_STATES = 1
            
            STATES_PER_STEP = 1 
            STEPS_PER_SEMITONE=1
            NUMBER_SEMITONES=1
            
            
            note_state_space = NoteStateSpace(NUMBER_SEMITONES, STEPS_PER_SEMITONE, STATES_PER_STEP )
            with_dummy = 1
            self.note_transition_model = NoteTransitionModel(note_state_space, with_dummy) 
        
        def test_values(self):
            self.assertTrue(len(self.note_transition_model.to_states) == 
                            len(self.note_transition_model.prev_states) ==
                            len(self.note_transition_model.probs))
            self.assertTrue(max(self.note_transition_model.to_states) < self.note_transition_model.num_states)
            self.assertTrue(max(self.note_transition_model.prev_states) < self.note_transition_model.num_states)
        

            
 
class TestBarNoteTransitionModelClass(unittest.TestCase):

    def setUp(self):
        self.bss = BarStateSpace(1, 1, 4)
        self.btm = BarTransitionModel(self.bss, 100)
        self.NUM_NOTE_STATES = 1
        note_state_space = NoteStateSpace(1,1,1)
        note_transition_model = NoteTransitionModel(note_state_space, 0)
        self.bntm = BarNoteTransitionModel(self.btm, note_transition_model) 

    def test_types(self):
         
        self.assertIsInstance(self.bntm, BarNoteTransitionModel)
        self.assertIsInstance(self.bntm, TransitionModel)
         
        self.assertIsInstance(self.bntm.states, np.ndarray)
        self.assertIsInstance(self.bntm.pointers, np.ndarray)
        self.assertIsInstance(self.bntm.probabilities, np.ndarray)
        self.assertIsInstance(self.bntm.log_probabilities, np.ndarray)
        self.assertIsInstance(self.bntm.num_states, int)
        self.assertIsInstance(self.bntm.num_transitions, int)
         
        self.assertTrue(self.bntm.states.dtype == np.uint32)
        self.assertTrue(self.bntm.pointers.dtype == np.uint32)
        self.assertTrue(self.bntm.probabilities.dtype == np.float)
        self.assertTrue(self.bntm.log_probabilities.dtype == np.float)

#     def test_values(self):
#         bss = BarStateSpace(1, 1, 4)
#         btm = BarTransitionModel(bss, 100)
#         
#         note_state_space = NoteStateSpace(1)
#         note_transition_model = NoteTransitionModel(note_state_space)
#         bntm = BarNoteTransitionModel(btm, note_transition_model) 
#         
#         self.assertTrue(np.allclose(bntm.prev_joint_idx,
#                                     [1, 3, 4, 6, 7, 8, 0, 2, 5, 5, 5, 9, 9]))
#         self.assertTrue(np.allclose(bntm.to_joint_idx, 
#                                     [2, 4, 5, 7, 8, 9, 0, 1, 1, 3, 6, 3, 6]))
# #         self.assertTrue(np.allclose(tm.probabilities,
# #                                     [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1,
# #                                      1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]))
# #         self.assertTrue(np.allclose(tm.log_probabilities,
# #                                     [0, 0, -33.33333, 0, 0, -25,
# #                                      0, 0, -33.33333, 0, 0, 0, 0,
# #                                      0, 0, -33.33333, 0, 0, -25,
# #                                      0, 0, -33.33333, 0, 0, 0, 0]))
#         self.assertTrue(btm.num_states == 10)
#         self.assertTrue(btm.num_transitions == 13)
    
    def test_compatibility(self):
        '''
        test if the logic of combining the existing BarTransition  with 
        NoteTransition  is correct. 
        When number of notes is 1, 
        the joint BarNoteTransitionModel transition should be equivalent to BarTransitionModel
        '''

        
#         joint BarNoteTransitionModel transition should be equivalent to BarTransitionModel
        if self.NUM_NOTE_STATES == 1:
            self.assertTrue(np.allclose(self.bntm.prev_joint_idx, self.btm.prev_states))
            self.assertTrue(np.allclose(self.bntm.to_joint_idx, self.btm.to_states))
            self.assertTrue(np.allclose(self.bntm.probs, self.btm.probs))
        elif self.NUM_NOTE_STATES == 2:
            expected_prev_idx = np.hstack((self.btm.prev_states, self.btm.prev_states + self.bss.num_states ))
            self.assertTrue(np.allclose(self.bntm.prev_joint_idx, expected_prev_idx))
#         self.assertTrue(np.allclose(tm.log_probabilities,
#                                     [0, 0, -33.33333, 0, 0, -25,
#                                      0, 0, -33.33333, 0, 0, 0, 0,
#                                      0, 0, -33.33333, 0, 0, -25,
#                                      0, 0, -33.33333, 0, 0, 0, 0]))
