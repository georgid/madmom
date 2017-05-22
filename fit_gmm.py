#!/usr/bin/env python
# encoding: utf-8
"""
This file contains code to fit the GMMs for the downbeat/beat tracker
described in
    "Rhythmic Pattern Modelling For Beat and Downbeat Tracking in Musical
         Audio"
    Florian Krebs, Sebastian Böck, and Gerhard Widmer
    Proceedings of the 14th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.


@author: Florian Krebs <florian.krebs@jku.at>

"""
from itertools import izip
import numpy as np
import argparse
import os
from madmom.ml.gmm import GMM
import cPickle
import sklearn.cluster as clst
import pylab as plt


def data_by_bars(file_list, allowed_meters=None, exclude_files=[]):
    """
    This function collects the specified features and organises them into
    bars/beats and positions within the bar/beat.

    :param file_list:           list of feature files
    :return: feat_from_bar:     list [n_bars] of np.arrays
                                [beats, subdivision, feat_dim]
    :return: file_from_bar      list of file ids corresponding to the
                                input file_list for each bar [n_bars]
    """
    feat_from_bar = []
    file_from_bar = []
    nbeats_from_bar = []
    # remove files that should be excluded
    file_list = remove_files(file_list, exclude_files)
    for i_file, feat_fn in enumerate(file_list):
        # load data
        data = np.load(feat_fn)
        beats = data['beats']
        # numpy array of features in beat sync
        # [num_beats x beat_subdivision x feat_dim]
        features = data['beat_features']
        beat_div = data['beat_div']
        beat_subdivision = features.shape[1]
        # beats_per_bar = data['beats_per_bar']
        if (beats.shape[0] - 1) != (len(beat_div) / beat_subdivision):
            raise ValueError('Beats and beat divisions mismatch')
        downbeat_idx = np.nonzero((beats[:, 1] == 1))[0]
        num_bars = sum(beats[:, 1] == 1) - 1
        for i_bar in range(num_bars):
            beats_per_bar = int(downbeat_idx[i_bar+1] -
                                downbeat_idx[i_bar])
            if beats_per_bar in allowed_meters:
                bar_start_div = downbeat_idx[i_bar]
                bar_end_div = downbeat_idx[i_bar+1]
                feat_from_bar.append(features[bar_start_div:bar_end_div, :, :])
                # save file id for corresponding bar
                file_from_bar.append(i_file)
                nbeats_from_bar.append(beats_per_bar)
    return feat_from_bar, np.array(file_from_bar), np.array(nbeats_from_bar)


def remove_files(file_list, exclude_files):
    """
    Remove all files that are in the list exclude_files from file_list

    :param file_list:     list of feature files
    :param exclude_files: list of files/strings to be excluded.
    :return: good_files   filtered list of feature files
    """
    good_files = []
    for file in file_list:
        remove = False
        for e_file in exclude_files:
            if e_file in file:
                remove = True
                break
        if not remove:
            good_files.append(file)
    return good_files


def get_bar_positions(beats_file, fps):
    """
    This function computes the beat position (float between 0 and
    num_beats_per_bar - 1) per frame

    :param beats_file:          beat and bar annotations
    :param fps:                 frames per second
    :return: beat_positions      for each frame, a beat position is given
    """
    # load beat annotations
    beats = np.loadtxt(beats_file)
    # set up frame vector
    beat_positions = np.empty((np.ceil(beats[-1, 0] * fps)))
    beat_positions.fill(np.nan)
    downbeats = np.where(beats[:, 1] == 1)[0]
    if beats[0, 1] != 1:
        # deal with pickup
        downbeats = np.insert(downbeats, 0, 0)
    if beats[-1, 1] != 1:
        # deal with non-complete bars at the end
        downbeats = np.append(downbeats, len(beats[:, 0])-1)
    for i_db, i_beat in np.ndenumerate(downbeats[0:-1]):
        next_downbeat_id = downbeats[i_db[0]+1]
        bar_start_frame = np.floor(beats[i_beat, 0] * fps)
        next_bar_start_frame = np.floor(beats[next_downbeat_id, 0] * fps)
        # set up array with time in sec of each frame center (first
        # frame center lies at 1/(2*fps)
        bar_frames_sec = np.arange(
            bar_start_frame, next_bar_start_frame) / fps + 1. / (2 * fps)
        beat_times = beats[i_beat:(next_downbeat_id+1), 0]
        # copy part of beats array to modify it later
        bar_pos_of_beats = np.copy(beats[i_beat:(next_downbeat_id+1), 1])
        # replace downbeat of next bar by last beat + 1 for interpolating
        bar_pos_of_beats[-1] = bar_pos_of_beats[-2] + 1
        bar_pos_of_frames = np.interp(bar_frames_sec, beat_times,
                                      bar_pos_of_beats)
        beat_positions[np.arange(int(bar_start_frame),
                       int(next_bar_start_frame))] = bar_pos_of_frames
    return beat_positions - 1


def distribute_cluster_among_meters(meteridx_from_bar, meters, n_clusters):
    # determine number of clusters per time signature
    meter_counts = np.bincount(meteridx_from_bar)
    num_meters = len(meters)
    temp = np.linspace(0, sum(meter_counts), n_clusters+1)[1:]
    mc = np.cumsum(meter_counts)
    cluster_dist = np.zeros(num_meters, dtype=int)
    for i_m in range(num_meters):
        cluster_dist[i_m] = sum((temp <= mc[i_m])) - sum(cluster_dist[:i_m])
    if sum(cluster_dist) != n_clusters:
        raise ValueError('Wrong distribution of clusters over meters!')
    return cluster_dist


def clustering_from_features(feat_from_bar, nbeats_from_bar, n_clusters,
                             save_fig_path='/tmp/'):
    """
    This function clusters the bars/beats according to the feature values.

    :param feat_from_bar:       list [n_bars] of np.arrays
                                 [beats, subdivision, feat_dim]
    :param nbeats_from_bar
    :param n_clusters:          number of clusters
    :param cluster_method:      how to cluster bars/beats {kmeans}
    :return: cluster_from_bar:  numpy array of cluster ids, one for each bar
                                in the training data (n_bars, )

    """

    meters, meteridx_from_bar = np.unique(nbeats_from_bar, return_inverse=True)
    nc = distribute_cluster_among_meters(meteridx_from_bar, meters, n_clusters)
    cluster_from_bar = np.ones(nbeats_from_bar.shape, dtype=int) * (-1)
    nbeats_from_cluster = np.ones(n_clusters, dtype=int) * (-1)
    centers = []
    for i_m in range(len(meters)):
        # extract data with current meter
        current_meter = (meteridx_from_bar == i_m)
        # reshape to (n_bars, n_dim * n_div_per_bar)
        data = reshape_data(feat_from_bar, filter=current_meter)
        # do clustering
        if nc[i_m] > 0:
            km = clst.KMeans(n_clusters=nc[i_m])
            km.fit(data)
            nbeats_from_cluster[range(0, nc[i_m]) + np.max(cluster_from_bar) +
                                1] = meters[i_m]
            cluster_labels = km.labels_ + np.max(cluster_from_bar) + 1
            cluster_from_bar[current_meter] = cluster_labels
            centers += list(km.cluster_centers_)

    if centers is not []:
        # exclude clusters with the wrong meter
        valid_clusters = (cluster_from_bar >= 0)
        cluster_counts = np.bincount(cluster_from_bar[valid_clusters])
        plot_clusters(centers, feat_from_bar[0].shape[2], cluster_counts,
                      save_fig_path=save_fig_path)
    return cluster_from_bar, nbeats_from_cluster


def plot_clusters(centers, feat_bands, cluster_counts, save_fig_path):
    plot_cols = np.round(np.sqrt(len(centers)))
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    max_feat = -1
    min_feat = 1
    # for c in centers:
    #     max_feat = np.max(np.append(c.flatten(), max_feat))
    #     min_feat = np.min(np.append(c.flatten(), min_feat))
    for i, c in enumerate(centers):
        max_feat = np.max(c.flatten())
        min_feat = np.min(c.flatten())
        plt.subplot(int(np.ceil(len(centers)/float(plot_cols))), plot_cols,
                    i+1)
        feat_len = len(c) / feat_bands
        for b in range(feat_bands):
            plt.plot(c[b*feat_len:(b+1)*feat_len]+max_feat*b)
        plt.title('cluster %i: (%i points)' % (i, cluster_counts[i]))
        plt.xlim(0, feat_len)
        plt.ylim(min_feat, max_feat*feat_bands)
    outfile = os.path.join(save_fig_path, 'out.png')
    print('writing patterns to {0}'.format(outfile))
    plt.savefig(outfile, figsize=(8, 8), dpi=300)
    return


def reshape_data(feat_from_bar, filter=None):
    """
    This prepares (formats) data for clustering
    :param feat_from_bar:       list [n_bars] of np.arrays
                                 [beats, subdivision, feat_dim]
    :returns data:              convert data shape to (n_bars, n_dim *
                                n_beats_per_bar)

    """
    # filter out specific meters
    if filter is not None:
        data = [feat_from_bar[x] for x in range(len(feat_from_bar))
                if filter[x]]
        feat_from_bar = data
    feat_beats = feat_from_bar[0].shape[0]  # number of beats per bar
    feat_div = feat_from_bar[0].shape[1]  # beat subdivision
    feat_bands = feat_from_bar[0].shape[2]  # feature dimension
    feat_len = feat_div * feat_beats
    num_bars = len(feat_from_bar)
    temp = np.array(feat_from_bar)
    # put into numpy array of shape=(n_bars, n_dim * n_div_per_bar)
    data = np.empty((num_bars, feat_len * feat_bands)) * np.nan
    for bar in range(num_bars):
        for fb in range(feat_bands):
            data[bar, fb*feat_len:(fb+1)*feat_len] = \
                temp[bar, :, :, fb].flatten('C')
    return data


def clustering_from_annotations(beat_files, meter_files, cluster_method):
    """
    This function clusters the bars/beats according to annotations.

    :param beat_files:                list of files containing beat annotations
    :param meter_files:               list of files containing meter annotations
    :param cluster_method:            how to cluster bars/beats {meter, rhythm,
                                      none}
    :return rhythm_from_bar           list of rhythm ids for each bar

    """
    pattern_id = 0
    rhythm_from_bar = []
    file_from_bar = []
    meter_from_rhythm = []
    rhythm_names = []
    nbars_from_file = []
    # TODO: What is this IF for? How do other cluster_methods work?
    if cluster_method == 'meter':
        meter_from_rhythm = []
    for f_id, (beat_fln, meter_fln) in enumerate(izip(beat_files, meter_files)):
        beats = np.loadtxt(beat_fln)
        # load meter annotation
        with open(meter_fln, 'r') as f:
            m = f.readline().strip()
        # remove slash and convert to int
        meter = map(int, m.split('/', 1))
        num_bars, _ = num_complete_bars(beats, meter[0])
        # save mapping from bar id to file id
        file_from_bar.extend([f_id] * num_bars)
        # save number of complete bars of current file
        nbars_from_file.append(num_bars)
        if cluster_method == 'meter':
            if meter_from_rhythm == []:
                # store meter
                meter_from_rhythm.append(meter)
                # add a name for this rhythm
                rhythm_names.append('/'.join(map(str, meter)))
                # add a pattern id for each bar
                rhythm_from_bar.extend([pattern_id] * num_bars)
            else:
                pattern_id = len(meter_from_rhythm)
                # check if this meter is already assigned to a pattern_id
                meter_new = 1
                for i, m in enumerate(meter_from_rhythm):
                    if (meter[0] == m[0]) and (meter[1] == m[1]):
                        pattern_id = i
                        meter_new = 0
                        break
                if meter_new:
                    meter_from_rhythm.append(meter)
                    # add a name for this rhythm
                    rhythm_names.append('/'.join(map(str, meter)))
                # add a pattern id for each bar
                rhythm_from_bar.extend([pattern_id] * num_bars)

    return meter_from_rhythm, file_from_bar, rhythm_from_bar, rhythm_names

GMM_MIXTURES = 4
GMM_INITIALISATIONS = 20


def fit_gmms(feat_from_bar_and_gmm, rhythm_from_bar):
    """
     This function fits a Gaussian Mixture Model (GMM) to each rhythm and bar
     position.

     :param feat_from_bar_and_gmm:       list [n_bars] of np.arrays
                                            [beats, subdivision, feat_dim]
     :param rhythm_from_bar:             list of rhythm ids for each bar
     :return: gmms                       [n_rhythms][bar_positions]

     """

    rhythms = np.unique(rhythm_from_bar[rhythm_from_bar >= 0])
    num_rhythms = len(rhythms)
    # number of beat subdivisions (HMM substates)
    num_substates = feat_from_bar_and_gmm[0].shape[1]
    gmms = [None] * num_rhythms
    for ri in rhythms:
        bar_idx = np.where(rhythm_from_bar == ri)[0]
        data_ri = np.array([feat_from_bar_and_gmm[x] for x in bar_idx])
        num_beats = feat_from_bar_and_gmm[bar_idx[0]].shape[0]
        gmms[ri] = [None] * num_beats
        for state in range(num_beats):
            gmms[ri][state] = [None] * num_substates
            for substate in range(num_substates):
                # make data (n, n_features)
                data = data_ri[:, state, substate, :]
                if data.ndim > 2:
                    data = np.squeeze(data)
                gmms[ri][state][substate] = GMM(n_components=GMM_MIXTURES,
                                                covariance_type='full')
                gmms[ri][state][substate].fit(data, n_iter=200,
                                              n_init=GMM_INITIALISATIONS)
    return gmms


def num_complete_bars(beats, num_beats_per_bar=None, verbose=1,
                      tolerance_beat_period_ratio=2.5):
    """
    This function computes the number and location of complete bars in the
    annotation. A bar is complete if there exists an annotation for all of its
    beats and if the following downbeat is annotated as well. Additionally,
    the interval between two successive beats is not allowed to be bigger than
    tolerance_beat_period_ratio times the beat period, in order to exclude
    pauses.

    :param beats:               beat/beat_type annotations [num_beats, 2]
    :param num_beats_per_bar:   number of beats per bar. If set, only bars
                                with this meter are returned as complete
                                bars. If none, all bars are returned.
    :param tolerance_beat_period_ratio: beat_period_ratio which is still
                                    considered as expressive. Above this ratio,
                                    a pause is assumed and the corresponding
                                    bar is excluded [default=2.5]
    :param verbose:             be verbose [default=0]

    :return num_bars            number of complete bars
    :return downbeat_idx        [B x 2] indices of downbeats that belong to
                                complete bar. The first index is the first
                                downbeat and the second is the downbeat of
                                the following bar.

    """
    beat_type = np.copy(beats[:, 1])
    # find downbeat indices
    downbeats = np.where(beats[:, 1] == 1)[0]
    num_bars = 0
    downbeat_idx = []
    for i_db in range(len(downbeats)-1):
        num_beats_i = downbeats[i_db+1] - downbeats[i_db]
        # check if meter is given and exclude the bar if it does not match
        if (num_beats_per_bar != None) and (num_beats_i != num_beats_per_bar):
            continue
        # check if beat types are increasing
        beat_ids = np.arange(downbeats[i_db], downbeats[i_db+1]+1)
        btype_diff = np.diff(beat_type[beat_ids[:-1]]).astype(int)
        if any(btype_diff != 1):
            continue
        # check for pauses
        beat_intervals = np.diff(beats[beat_ids, 0])
        deviation_ratio = beat_intervals[1:] / beat_intervals[0:-1]
        if any(deviation_ratio > tolerance_beat_period_ratio):
            if verbose:
                print 'Warning fit_gmm.py: pause(s) detected'
            continue
        # if all checks passed, save bar
        num_bars += 1
        downbeat_idx.append([downbeats[i_db], downbeats[i_db+1]])
    return num_bars, downbeat_idx


def save_gmms(gmms, nbeats_from_cluster, means, stds, model_folder='/tmp'):
    """
    This function saves the GMMs to file.

    :param gmms:                concatenated list of GMM objects
                                gmms[pattern][state][substate]
    :param nbeats_from_cluster: number of beats per pattern.
    :param model_folder:        location where model files are stored
    """
    for i_g, g in enumerate(gmms):
        # load features from numpy binary format
        outfile_pickle = os.path.join(
            model_folder, 'gmm_pattern_' + str(nbeats_from_cluster[i_g]) + '_' +
            str(i_g) + '.pkl')
        save_data = {'gmms': g, 'num_beats': nbeats_from_cluster[i_g],
                     'feat_means': means, 'feat_stds': stds}
        cPickle.dump(save_data, open(outfile_pickle, 'w'))
        print "Save GMMs to %s" % outfile_pickle


def normalise_features(feat_from_bar):
    """
    This function normalises the features to zero mean and unit variance

    :param feat_from_bar:       list of numpy arrays, each [num_beats x
                                subdivisions x feat dimensions]
    """
    feat_bands = feat_from_bar[0].shape[2]
    means = np.empty(feat_bands)
    stds = np.empty(feat_bands)
    for fb in range(feat_bands):
        feats = [y for x in feat_from_bar for y in x[:, :, fb].flatten()]
        means[fb] = np.mean(feats)
        stds[fb] = np.std(feats)
    feats_norm = [(x - means[np.newaxis, np.newaxis, :]) /
            stds[np.newaxis, np.newaxis, :] for x in feat_from_bar]
    return feats_norm, means, stds


def main():
    """fit_gmms"""

    cluster_methods = ['kmeans', 'meter', 'rhythm', 'none']

    # define parser
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description='''
    This program fits GMMs to features of a training dataset to be used for
    downbeat tracking. The input files have to contain a dictionary with the
    keys 'beats', 'beat_features' (numpy array of features in beat sync [
    num_beats x beat_subdivision x feat_dim], and 'beat_div'.

    ''')
    # add arguments
    p.add_argument('-f', dest='files', nargs='+', help='files to be processed')
    p.add_argument('-e', dest='exclude_files', default=None,
                   help='text file with files that are excluded from the'
                        'training')
    p.add_argument('-c', dest='cluster_method', default='kmeans',
                   choices=cluster_methods, help='clustering method '
                   '[default=%(default)s]', action='store')
    p.add_argument('-nc', dest='n_clusters', default=2, help='number of '
                   'clusters [default=%(default)u]', action='store', type=int)
    p.add_argument('-a', dest='allowed_meters', nargs='+', type=int,
                   help='allowed numerators of the time signature')
    p.add_argument('-o', dest='out_folder', default='/tmp',
                   help='folder where model files are stored '
                   '[default=%(default)s]', action='store')
    # parse arguments
    args = p.parse_args()
    # read list of training files from text file
    if args.exclude_files is None:
        exclude_files = []
    else:
        try:
            # remove newline from string
            exclude_files = [line.strip() for line in
                             open(args.exclude_files, 'r')]
        except IOError:
            print "Sorry,", args.exclude_files, "does not exist"
    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)
    
    # Organise feature into bars and gmms
    feat_from_bar, file_from_bar, nbeats_from_bar = data_by_bars(
        args.files, allowed_meters=args.allowed_meters,
        exclude_files=exclude_files)
    # Normalise features to zero-mean and unit-variance
    feat_from_bar_norm, means, stds = normalise_features(feat_from_bar)
    # Perform clustering of the bars
    if args.cluster_method == 'kmeans':
        cluster_data, nbeats_from_cluster = clustering_from_features(
            feat_from_bar_norm, nbeats_from_bar, n_clusters=args.n_clusters,
            save_fig_path=args.out_folder)
    elif args.cluster_method in ['meter', 'rhythm', 'none']:
        cluster_data = clustering_from_annotations(args.files,
                                                   args.cluster_method)
    else:
        raise ValueError('The specified cluster_method does not exist!')
    # Fit GMMs to the clustered data
    gmms = fit_gmms(feat_from_bar, cluster_data)
    if args.out_folder is not None:
        # Save GMMs to file
        save_gmms(gmms, nbeats_from_cluster, means, stds,
                  model_folder=args.out_folder)

if __name__ == "__main__":
    main()
