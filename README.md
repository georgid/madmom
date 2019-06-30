# madmom (with synchronous note onsets and beat position)

This fork adds a hidden state of a musical note (in short note state). 
It is however generic enough and can be used with any other musical concept as hidden state. 
For this purpose the NoteTransitionModel and and GMMNotePatternTrackingObservationModel classes are dummy

The joint bar+note state space is a Cartesian product of the existing BarStateSpace and the new NoteStateSpace. 
 
NOTE: There is no class for joint BarNoteStateSpace, the Cartesian multiplication is handled inside the BarNoteTransitionModel and GMMNotePatternTrackingObservationModel

NOTE: for the sake of simplicity, it is implemented to work with only [one rhythmic pattern](madmom.models.__init__.PATTERNS_BALLROOM). 

Modifications to the original code:
- [saving class variables](https://github.com/georgid/madmom/blob/master/madmom/features/beats_hmm.py#L394)

Usage
--------
bin/GMMNotePatternTracker

--------------------------------------------------

## Citation

> Georgi Dzhambazov, André Holzapfel, Ajay Srinivasamurthy, Xavier Serra, 
> Metrical-Accent Aware Vocal Onset Detection in Polyphonic Audio, In Proceedings of ISMIR 2017

NOTE: This repository works in together with the other repository based on [pypYIN](https://github.com/georgid/pypYIN/) for pitch tracking.

---------------------
The rest of the documentation about [Madmom in general](https://github.com/georgid/madmom)
