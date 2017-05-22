Run downbeat detection:
------------------------------

# install. see instructions at https://github.com/CPJKU/madmom/

# detect beats one recordin. write output to file
python ~/workspace/madmom/bin/GMMPatternTracker single  -o $OUTPUT_FILE  $INPUT_AUDIO

#  detect beats one recording
python ~/workspace/madmom/bin/GMMPatternTracker single $INPUT_AUDIO



Run downbeat detection + note onset detection (slower): 
------------------------------
# install. see instructions at
https://github.com/georgid/madmom/

# set path
PATH_DATASET=/Users/joro/workspace/otmm_vocal_segments_dataset/

set WITH_NOTES_STATES=0 # to have no notes-influence
OUTPUT_FILE=$PATH_DATASET/experiments/ht_0_0058/$MBID/$MBID.estimatedbeats
INPUT_AUDIO=$PATH_DATASET/data/$MBID/$MBID.wav

# detect beats one recording:
python ~/workspace/madmom_notes/bin/GMMNotePatternTracker single  -o $OUTPUT_FILE  $INPUT_AUDIO

# or many: modify suffix in file
python ~/workspace/madmom_notes/bin/GMMNotePatternTracker_all.py



Eval estimated beats: 
----------------------------------------------------


### convert , to \t if needed 
for i in `ls $PATH_DATASET/data/*/*.beats`; do tr ',' '\t' <$i >${i}_tab; done

# get list of detected files:
for i in `ls $PATH_DATASET//experiments/ht_0_0058/*/*.estimatedbeats_norm`; do echo $i; done


# eval  beats ( skipping first 5 seconds) 
python ~/workspace/madmom_notes/bin/evaluate beats -i --ann_dir "/Users/joro/workspace/otmm_vocal_segments_dataset/data/" -a .beats_tab --det_dir "/Users/joro/workspace/otmm_vocal_segments_dataset/experiments/ht_0_0058/" -d .estimatedbeats --skip 5	
