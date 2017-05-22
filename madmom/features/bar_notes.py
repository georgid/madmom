# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
import numpy as np

from madmom.features.bar_note_observation import GMMNoteObservationModel,\
    GMMNotePatternTrackingObservationModel
from madmom.Params import  USUL_NUM_BEATS
from pypYIN.MonoNoteParameters import WITH_NOTES_STATES, WITH_MAKAM
import os



'''
Created on Feb 26, 2017

@author: joro
'''

from madmom.processors import Processor
from madmom.features.bar_notes_hmm import NoteStateSpace, NoteTransitionModel,\
    BarNoteStateSpace, BarNoteTransitionModel
from demo import store_results, determine_file_with_extension
            
from pypYIN.MonoNoteParameters import STEPS_PER_SEMITONE, NUM_SEMITONES


# class for pattern tracking
class NotePatternTrackingProcessor(Processor):
    """
    Note and Pattern tracking with a dynamic Bayesian network (DBN) approximated by a
    Hidden Markov Model (HMM).

    Parameters
    ----------
    pattern_files : list
        List of files with the patterns (including the fitted GMMs and
        information about the number of beats).
    min_bpm : list, optional
        Minimum tempi used for pattern tracking [bpm].
    max_bpm : list, optional
        Maximum tempi used for pattern tracking [bpm].
    num_tempi : int or list, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacings, otherwise a linear spacings.
    transition_lambda : float or list, optional
        Lambdas for the exponential tempo change distributions (higher values
        prefer constant tempi from one beat to the next .one)
    downbeats : bool, optional
        Report only the downbeats instead of the beats and the respective
        position inside the bar.
    fps : float, optional
        Frames per second.


    """
    # TODO: this should not be lists (lists are mutable!)
    MIN_BPM = [100, 90, 180]
    MAX_BPM = [150, 250, 250]
    NUM_TEMPI = [None, None, None]
    # TODO: make this parametric
    # Note: if lambda is given as a list, the individual values represent the
    #       lambdas for each transition into the beat at this index position
    TRANSITION_LAMBDA = [100, 100, 100]
    
    if not WITH_MAKAM:   
        TRANSITION_LAMBDA = [100, 100]
        NUM_TEMPI = [None, None]
    
    # around min and max for makam dataset
#     MIN_BPM = [100] 
#     MAX_BPM = [205] 
#     NUM_TEMPI = [None]
#     # TODO: make this parametric
#     # Note: if lambda is given as a list, the individual values represent the
#     #       lambdas for each transition into the beat at this index position
#     TRANSITION_LAMBDA = [100]
    
    def __init__(self, pattern_files, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                 downbeats=False, fps=None, usul_type=None, **kwargs):
        # pylint: disable=unused-argument
        # pylint: disable=no-name-in-module

        import pickle
        from .beats_hmm import (BarStateSpace, BarTransitionModel,
                                MultiPatternStateSpace,
                                MultiPatternTransitionModel,
                                GMMPatternTrackingObservationModel)
        from ..ml.hmm import HiddenMarkovModel as Hmm

        # expand num_tempi and transition_lambda to lists if needed
        if not isinstance(num_tempi, list):
            num_tempi = [num_tempi] * len(pattern_files)
        if not isinstance(transition_lambda, list):
            transition_lambda = [transition_lambda] * len(pattern_files)
        # check if all lists have the same length
        if not (len(min_bpm) == len(max_bpm) == len(num_tempi) ==
                len(transition_lambda) == len(pattern_files)):
            raise ValueError('`min_bpm`, `max_bpm`, `num_tempi` and '
                             '`transition_lambda` must have the same length '
                             'as number of patterns.')
        # save some variables
        self.downbeats = downbeats
        self.fps = fps
        self.num_beats = []

        # convert timing information to construct a state space
        min_interval = 60. * self.fps / np.asarray(max_bpm)
        max_interval = 60. * self.fps / np.asarray(min_bpm)
        # collect beat/bar state spaces, transition models, and GMMs
        state_spaces = []
        transition_models = []
        gmms = []
        # check that at least one pattern is given
        if not pattern_files:
            raise ValueError('at least one rhythmical pattern must be given.')
        
        # the note state space and transition model are same for all patterns
        if WITH_NOTES_STATES:

            self.note_state_space = NoteStateSpace(NUM_SEMITONES, STEPS_PER_SEMITONE, 3 )
        else:            
            self.note_state_space = NoteStateSpace(1, 1, 1)
        note_transition_model = NoteTransitionModel(self.note_state_space, usul_type, WITH_NOTES_STATES)

        # load the patterns
        for p, pattern_file in enumerate(pattern_files):
            with open(pattern_file, 'rb') as f:
                # Python 2 and 3 behave differently
                # TODO: use some other format to save the GMMs (.npz, .hdf5)
                try:
                    # Python 3
                    pattern = pickle.load(f, encoding='latin1')
                except TypeError:
                    # Python 2 doesn't have/need the encoding
                    pattern = pickle.load(f)
            # get the fitted GMMs and number of beats
            num_beats = pattern['num_beats']
            if USUL_NUM_BEATS[usul_type] != num_beats: # workaround to use only the pattern for the usul of the current recording
                continue
            gmms.append(pattern['gmms'])
            self.num_beats.append(num_beats)
            # model each rhythmic pattern as a bar
            bar_state_space = BarStateSpace(num_beats, min_interval[p],
                                        max_interval[p], num_tempi[p])
            bar_transition_model = BarTransitionModel(bar_state_space,
                                                  transition_lambda[p])

            bar_note_transition_model = BarNoteTransitionModel(bar_transition_model, note_transition_model, WITH_NOTES_STATES)
                                                                
            state_spaces.append(bar_state_space) # in fact only one state space
            transition_models.append(bar_note_transition_model)
        
        # create multi pattern state space, transition and observation model
        self.st = MultiPatternStateSpace(state_spaces)
        self.tm = MultiPatternTransitionModel(transition_models)
        
        note_om = GMMNoteObservationModel(self.note_state_space)
        
        bar_om = GMMPatternTrackingObservationModel(gmms, self.st)
        
        self.bar_note_om = GMMNotePatternTrackingObservationModel(bar_om, note_om)
        # instantiate a HMM
        self.hmm = Hmm(self.tm, self.bar_note_om, None)
    
    def set_file_name(self, file_name):
            self.file_name = file_name
    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
    
    def process(self, activations):
        """
        Detect the beats based on the given activations.

        Parameters
        ----------
        activations : numpy array shape (observations, 3)
            dimensions  :,0:1 SpectraL Flux Activations (i.e. multi-band spectral features).
            dimension   :,2:3 Note pitch + prob.  
        Returns
        -------
        beats : numpy array
            Detected beat positions [seconds].

        """
        # get the best state path by calling the viterbi algorithm
        path, _ = self.hmm.viterbi(activations)
        num_bar_states = len(self.st.state_positions) # NOT SURE what happens with more than 1 pattern
        num_note_states = self.note_state_space.num_states
        
        #decompose combined state-space into bar and notes
        (path_indices_bar, path_indices_note) = np.unravel_index(path, (num_bar_states, num_note_states), order='F')
        
        # print decoded note states sequence
        f0_values = activations[:,2]
        onsetframes  = notestates_to_onsetframes( path_indices_note, f0_values)
        hop_time = float (1.0 / self.fps)
        # store detected timestamps
        
            
        
        MBID = os.path.basename(self.file_name)[:-4]
        
        extension = determine_file_with_extension(NUM_SEMITONES, STEPS_PER_SEMITONE, WITH_BEAT_ANNOS=0, WITH_DETECTED_BEATS=1)
        URI_output = os.path.join(self.output_dir, MBID + extension)
    
        store_results(onsetframes, URI_output, hop_time)
        
        
        
        # the positions inside the pattern (0..num_beats)
        positions = self.st.state_positions[path_indices_bar]
        # corresponding beats (add 1 for natural counting)
        beat_numbers = positions.astype(int) + 1
        # transitions are the points where the beat numbers change
        # FIXME: we might miss the first or last beat!
        #        we could calculate the interval towards the beginning/end to
        #        decide whether to include these points
        beat_positions = np.nonzero(np.diff(beat_numbers))[0] + 1
        # stack the beat positions (converted to seconds) and beat numbers
        beats = np.vstack((beat_positions / float(self.fps),
                           beat_numbers[beat_positions])).T
        # return the downbeats or beats and their beat number
        if self.downbeats:
            return beats[beats[:, 1] == 1][:, 0]
        else:
            return beats
        

    
    @staticmethod
    def add_arguments(parser, pattern_files=None, min_bpm=MIN_BPM,
                      max_bpm=MAX_BPM, num_tempi=NUM_TEMPI,
                      transition_lambda=TRANSITION_LAMBDA):
        """
        Add DBN related arguments for pattern tracking to an existing parser
        object.

        Parameters
        ----------
        parser : argparse parser instance
            Existing argparse parser object.
        pattern_files : list
            Load the patterns from these files.
        min_bpm : list, optional
            Minimum tempi used for beat tracking [bpm].
        max_bpm : list, optional
            Maximum tempi used for beat tracking [bpm].
        num_tempi : int or list, optional
            Number of tempi to model; if set, limit the number of states and
            use log spacings, otherwise a linear spacings.
        transition_lambda : float or list, optional
            Lambdas for the exponential tempo change distribution (higher
            values prefer constant tempi from one beat to the next one).

        Returns
        -------
        parser_group : argparse argument group
            Pattern tracking argument parser group

        Notes
        -----
        `pattern_files`, `min_bpm`, `max_bpm`, `num_tempi`, and
        `transition_lambda` must the same number of items.

        """
        from ..utils import OverrideDefaultListAction
        # add GMM options
        if pattern_files is not None:
            g = parser.add_argument_group('GMM arguments')
            g.add_argument('--pattern_files', action=OverrideDefaultListAction,
                           default=pattern_files,
                           help='load the patterns (with the fitted GMMs) '
                                'from these files (comma separated list)')
        # add HMM parser group
        g = parser.add_argument_group('dynamic Bayesian Network arguments')
        g.add_argument('--min_bpm', action=OverrideDefaultListAction,
                       default=min_bpm, type=float, sep=',',
                       help='minimum tempo (comma separated list with one '
                            'value per pattern) [bpm, default=%(default)s]')
        g.add_argument('--max_bpm', action=OverrideDefaultListAction,
                       default=max_bpm, type=float, sep=',',
                       help='maximum tempo (comma separated list with one '
                            'value per pattern) [bpm, default=%(default)s]')
        g.add_argument('--num_tempi', action=OverrideDefaultListAction,
                       default=num_tempi, type=int, sep=',',
                       help='limit the number of tempi; if set, align the '
                            'tempi with log spacings, otherwise linearly '
                            '(comma separated list with one value per pattern)'
                            ' [default=%(default)s]')
        g.add_argument('--transition_lambda', action=OverrideDefaultListAction,
                       default=transition_lambda, type=float, sep=',',
                       help='lambda of the tempo transition distribution; '
                            'higher values prefer a constant tempo over a '
                            'tempo change from one bar to the next one (comma '
                            'separated list with one value per pattern) '
                            '[default=%(default)s]')
        # add output format stuff
        g = parser.add_argument_group('output arguments')
        g.add_argument('--downbeats', action='store_true', default=False,
                       help='output only the downbeats')
        # return the argument group so it can be modified if needed
        return g

def notestates_to_onsetframes( path_indices_note, f0_values):
        '''
        Parameters
        ------------------
        
        path_indices_note: list (n) 
            indices of decoded note_states
        
        f0_values: n
            detected f0
        
        Returns
        -----------------
        frame numbers of onsets
        '''
        STEPS_PER_PITCH = 3
        step_states = np.mod(path_indices_note, STEPS_PER_PITCH) # convert to attack, sustain, sience.
        
        prev_IsVoiced = True
        onsetFrames = []
        for iFrame in range(len(step_states)):
                        
            isVoiced = step_states[iFrame] < STEPS_PER_PITCH - 1 and f0_values[iFrame] > 0
            
            if isVoiced and iFrame != len(step_states)-1: # sanity check                  
                if prev_IsVoiced == 0: # set onset at non-voiced-to-voiced transition
                    onsetFrames.append( iFrame )
            prev_IsVoiced = isVoiced
            
#         path_indices_note_phase < 3 # intersect these
# 
#         f0_values[2,:] > 0 
#         # zero non-voiced 1 voiced
#         
#         np.diff # only there where  changes occur.
#         
#         # get only changes with 0-> 1 
        return onsetFrames
