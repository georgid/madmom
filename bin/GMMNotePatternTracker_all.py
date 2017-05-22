'''
Created on Apr 11, 2017

@author: joro
'''
from GMMNotePatternTracker import main
import sys
import os

from doit_all import  ordered_list_MBID, create_output_dirs
from pypYIN.MonoNoteParameters import WITH_MAKAM


def extract_beats_all():
            if WITH_MAKAM:
                data_dir = '/Users/joro/workspace/otmm_vocal_segments_dataset/'
            else:
                data_dir = '/Users/joro/workspace/lakh_vocal_segments_dataset/'
                
            experiments_dir = create_output_dirs(data_dir, with_beat_annotations=False)
            
            for MBID in ordered_list_MBID.keys():
                MBID_dir  = os.path.join(experiments_dir, MBID)
               
                URI_output = os.path.join(MBID_dir, MBID + '.estimatedbeats')
                
                URI_wav = os.path.join(data_dir, 'data/', MBID, MBID + '.wav')
                bpm_gt = ordered_list_MBID[MBID][1]
                
                bpm_min = bpm_gt - 10
                bpm_max = bpm_gt + 10
                if WITH_MAKAM: # 3 usuls
                    bpm_min_str = str((bpm_min, bpm_min, bpm_min)).replace('(','').replace(')','')
                    bpm_max_str = str((bpm_max, bpm_max, bpm_max)).replace('(','').replace(')','')
                else: # 2: 3/4 and 4/4
                    bpm_min_str = str((bpm_min, bpm_min)).replace('(','').replace(')','')
                    bpm_max_str = str(( bpm_max, bpm_max)).replace('(','').replace(')','')


                sys.argv = ['dummy', '--min_bpm', bpm_min_str, '--max_bpm' , bpm_max_str , 'single','-o', URI_output, URI_wav ]
                main()

if __name__ == '__main__':
    extract_beats_all()