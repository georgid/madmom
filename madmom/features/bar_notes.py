# encoding: utf-8
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments


'''
Created on Feb 26, 2017

@author: joro
'''
import numpy as np

from madmom.processors import Processor
from madmom.features.bar_notes_hmm import NoteStateSpace, NoteTransitionModel,\
    BarNoteStateSpace, BarNoteTransitionModel, GMMNoteTrackingObservationModel,\
    GMMNotePatternTrackingObservationModel





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

    Notes
    -----
    `min_bpm`, `max_bpm`, `num_tempo_states`, and `transition_lambda` must
    contain as many items as rhythmic patterns are modeled (i.e. length of
    `pattern_files`).
    If a single value is given for `num_tempo_states` and `transition_lambda`,
    this value is used for all rhythmic patterns.

    Instead of the originally proposed state space and transition model for
    the DBN [1]_, the more efficient version proposed in [2]_ is used.

    References
    ----------
    .. [1] Florian Krebs, Sebastian B�ck and Gerhard Widmer,
           "Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical
           Audio",
           Proceedings of the 15th International Society for Music Information
           Retrieval Conference (ISMIR), 2013.
    .. [2] Florian Krebs, Sebastian B�ck and Gerhard Widmer,
           "An Efficient State Space Model for Joint Tempo and Meter Tracking",
           Proceedings of the 16th International Society for Music Information
           Retrieval Conference (ISMIR), 2015.

    Examples
    --------
    Create a PatternTrackingProcessor from the given pattern files. These
    pattern files include fitted GMMs for the observation model of the HMM.
    The returned array represents the positions of the beats and their position
    inside the bar. The position is given in seconds, thus the expected
    sampling rate is needed. The position inside the bar follows the natural
    counting and starts at 1.

    >>> from madmom.models import PATTERNS_BALLROOM
    >>> proc = PatternTrackingProcessor(PATTERNS_BALLROOM, fps=50)
    >>> proc  # doctest: +ELLIPSIS
    <madmom.features.beats.PatternTrackingProcessor object at 0x...>

    Call this PatternTrackingProcessor with a multi-band spectrogram to obtain
    the beat and downbeat positions. The parameters of the spectrogram have to
    correspond to those used to fit the GMMs.

    >>> from madmom.processors import SequentialProcessor
    >>> from madmom.audio.spectrogram import LogarithmicSpectrogramProcessor, \
SpectrogramDifferenceProcessor, MultiBandSpectrogramProcessor
    >>> log = LogarithmicSpectrogramProcessor()
    >>> diff = SpectrogramDifferenceProcessor(positive_diffs=True)
    >>> mb = MultiBandSpectrogramProcessor(crossover_frequencies=[270])
    >>> pre_proc = SequentialProcessor([log, diff, mb])

    >>> act = pre_proc('tests/data/audio/sample.wav')
    >>> proc(act)  # doctest: +ELLIPSIS
    array([[ 0.82,  4.  ],
           [ 1.78,  1.  ],
           ...,
           [ 3.7 ,  3.  ],
           [ 4.66,  4.  ]])
    """
    # TODO: this should not be lists (lists are mutable!)
    MIN_BPM = [55, 60]
    MAX_BPM = [205, 225]
    NUM_TEMPI = [None, None]
    # TODO: make this parametric
    # Note: if lambda is given as a list, the individual values represent the
    #       lambdas for each transition into the beat at this index position
    TRANSITION_LAMBDA = [100, 100]

    def __init__(self, pattern_files, min_bpm=MIN_BPM, max_bpm=MAX_BPM,
                 num_tempi=NUM_TEMPI, transition_lambda=TRANSITION_LAMBDA,
                 downbeats=False, fps=None, **kwargs):
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
            gmms.append(pattern['gmms'])
            num_beats = pattern['num_beats']
            self.num_beats.append(num_beats)
            # model each rhythmic pattern as a bar
            bar_state_space = BarStateSpace(num_beats, min_interval[p],
                                        max_interval[p], num_tempi[p])
            bar_transition_model = BarTransitionModel(bar_state_space,
                                                  transition_lambda[p])
            note_state_space = NoteStateSpace()
            note_transition_model = NoteTransitionModel(note_state_space)
            bar_note_state_space = BarNoteStateSpace(bar_state_space, note_state_space)
            
            bar_note_transition_model = BarNoteTransitionModel(bar_transition_model, note_transition_model)
            
            state_spaces.append(bar_state_space)
            transition_models.append(bar_note_transition_model)
        # create multi pattern state space, transition and observation model
        self.st = MultiPatternStateSpace(state_spaces)
        self.tm = MultiPatternTransitionModel(transition_models)
        
        note_om = GMMNoteTrackingObservationModel(note_state_space)
        
        bar_om = GMMPatternTrackingObservationModel(gmms, self.st)
        
        self.om = GMMNotePatternTrackingObservationModel(bar_om, note_om)
        # instantiate a HMM
        self.hmm = Hmm(self.tm, self.om, None)

    def process(self, activations):
        """
        Detect the beats based on the given activations.

        Parameters
        ----------
        activations : numpy array
            Activations (i.e. multi-band spectral features).
    
        Returns
        -------
        beats : numpy array
            Detected beat positions [seconds].

        """
        # get the best state path by calling the viterbi algorithm
#         path, _ = self.hmm.viterbi(activations, note_activations)
        path, _ = self.hmm.viterbi(activations)
        
        #TODO: decompose combined state-space 
        
        # the positions inside the pattern (0..num_beats)
        positions = self.st.state_positions[path]
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
