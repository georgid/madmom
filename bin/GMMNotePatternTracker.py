#!/usr/bin/env python
# encoding: utf-8
"""
GMMPatternTracker for tracking (down-)beats based on rhythmic patterns.

"""

from __future__ import absolute_import, division, print_function

import argparse

from madmom.processors import IOProcessor, io_arguments
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                      LogarithmicSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor,
                                      MultiBandSpectrogramProcessor)
from madmom.features import ActivationsProcessor
from madmom.models import PATTERNS_BALLROOM, PATTERNS_MAKAM
from madmom.features.bar_notes import NotePatternTrackingProcessor
from madmom.features.bar_note_observation import    CombinedFeatureProcessor
import mir_eval
import os
from demo import get_meter_from_rec
from doit_all import create_output_dirs
from pypYIN.MonoNoteParameters import WITH_MAKAM

import sys
from madmom.features.beats import PatternTrackingProcessor
from madmom.Params import WITH_GMM_PATTERN_TRACKER






def main():
    """GMMNotePatternTracker"""

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    The GMMPatternTracker program detects rhythmic patterns in an audio file
    and reports the (down-)beats according to the method described in:

    HERE Add Note Tracking description

    Instead of the originally proposed state space and transition model for the
    DBN, the following is used:

    "An Efficient State Space Model for Joint Tempo and Meter Tracking"
    Florian Krebs, Sebastian Böck and Gerhard Widmer.
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.



    ''')
    # version
    p.add_argument('--version', action='version',
                   version='GMMNotePatternTracker.2017')
    # add arguments
    io_arguments(p, output_suffix='.beats.txt')
    ActivationsProcessor.add_arguments(p)
    SignalProcessor.add_arguments(p, norm=False, gain=0)
    if WITH_GMM_PATTERN_TRACKER:
        PatternTrackingProcessor.add_arguments(p)
    else:
        NotePatternTrackingProcessor.add_arguments(p)

    # parse arguments
    args = p.parse_args()

    # set immutable defaults
    args.num_channels = 1
    args.sample_rate = 44100
    args.fps = 172
#     args.fps = 50
    args.num_bands = 12
    args.fmin = 30
    args.fmax = 17000
    args.norm_filters = False
    args.mul = 1
    args.add = 1
    args.diff_ratio = 0.5
    args.positive_diffs = True
    args.crossover_frequencies = [270]
    if WITH_MAKAM:
        args.pattern_files = PATTERNS_MAKAM
    else:
        args.pattern_files = PATTERNS_BALLROOM
    
    
    excerpt_URI = os.path.join(os.path.dirname(args.infile.name),  'excerpt.txt') # process frames until end_ts
    start_ts, end_ts = load_excerpt(excerpt_URI)
    
    rec_ID = os.path.basename(args.infile.name) [:-4]
    args.usul_type = get_meter_from_rec(rec_ID)
    
    # print arguments
    if args.verbose:
        print(args)

    # input processor
    if args.load:
        # load the activations from file
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        # define an input processor
        sig = SignalProcessor(**vars(args))
        frames = FramedSignalProcessor(**vars(args))
        filt = FilteredSpectrogramProcessor(**vars(args))
        log = LogarithmicSpectrogramProcessor(**vars(args))
        diff = SpectrogramDifferenceProcessor(**vars(args))
        mb = MultiBandSpectrogramProcessor(**vars(args))
        
        if WITH_GMM_PATTERN_TRACKER:
            in_processor = [sig, frames, filt, log, diff, mb]
        else:
            in_processor = CombinedFeatureProcessor([sig, frames, filt, log, diff, mb])
            pitch_framesize = 2048 # optimal for pitch size
            hopsize = int( float(args.sample_rate) / float(args.fps) )
            in_processor.set_params( args.infile.name, hopsize , pitch_framesize, end_ts )
        
    # output processor
    if args.save:
        # save the multiband features to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # downbeat processor
        if WITH_GMM_PATTERN_TRACKER:
            downbeat_processor = PatternTrackingProcessor(**vars(args))
        else:
            downbeat_processor = NotePatternTrackingProcessor(**vars(args))
            downbeat_processor.set_file_name(args.infile.name)
            data_dir = os.path.join(os.path.dirname(args.infile.name),os.pardir, os.pardir)
            output_dir = create_output_dirs(data_dir, with_beat_annotations=False)
            downbeat_processor.set_output_dir(os.path.join(output_dir,rec_ID))
        
        if args.downbeats:
            # simply write the timestamps
            from madmom.utils import write_events as writer
        else:
            # borrow the note writer for outputting timestamps + beat numbers
            from madmom.features.notes import write_notes as writer
        # sequentially process the features
        out_processor = [downbeat_processor, writer]

    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)
    
    print ( 'working on file {}...'.format( args.infile.name ) )
    
    # and call the processing function
    args.func(processor, **vars(args))
    
   

def load_excerpt(URI_excerpt):
    start_ts, end_ts, _ = mir_eval.io.load_delimited(URI_excerpt, [float,float,str],delimiter='\t')
    return float(start_ts[0]), float(end_ts[0])



if __name__ == '__main__':
    main()