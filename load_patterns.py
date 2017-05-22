'''
Created on Mar 30, 2017

script to load patterns from json
@author: joro
'''
import pickle
from madmom.ml.gmm import GMM
import json
import numpy as np

GMM_MIXTURES = 2
FEATURE_DIMENSION = 2
 
if __name__ == '__main__':
    
#     pattern = pickle.load(open('patterns/2013/ballroom_pattern_3_4.pkl'))
    gmm_model = json.load(open('/Users/joro/Dropbox/TurkishMakam_vocalDataset_training/results/observation_model.json'))
    
    for iUsul in range(len(gmm_model)):    
        usul_gmm_model = gmm_model[iUsul]
        pattern = {}
        num_models = usul_gmm_model['barGrid']

        num_beats = int(num_models / 8) # 8 positions per 8-th note
        pattern['gmms'] = []
        pattern['time_signature'] = [num_beats,8]
        pattern['num_beats'] = num_beats
        
        for i in range(num_models):
            gmm = GMM(n_components=GMM_MIXTURES,covariance_type='full')
    
            raw_covars = usul_gmm_model['gmmparams'][i]['covars']['_ArrayData_']    
            gmm.covars_ = np.array(raw_covars).reshape(FEATURE_DIMENSION, FEATURE_DIMENSION, GMM_MIXTURES)
            
            raw_means = usul_gmm_model['gmmparams'][i]['means']
            gmm.means_ =  np.array(raw_means).reshape(FEATURE_DIMENSION,  GMM_MIXTURES)
    # 
            gmm.weights_ = usul_gmm_model['gmmparams'][i]['mixprop']
            
            pattern['gmms'].append(gmm)
        pattern_URI = 'madmom/models/patterns/2017/makam_pattern_' + str(num_beats)  + '_8.pkl'
        pickle.dump(pattern, open(pattern_URI,'w'))