#!/usr/bin/env python3

import yaml
import rospy
import numpy as np
from processing_cvsa.msg import eeg_power
from rosneuro_msgs.msg import NeuroOutput
from scipy.spatial.distance import cdist # Per calcolare le distanze

class KMeansClassifier:
    def __init__(self):
        rospy.init_node('kmeans_classifier', anonymous=True)
        
        try:
            self.path_decoder = rospy.get_param('~path_kmeans_model')
        except KeyError as e:
            rospy.logfatal(f"Parametro mancante: {e}. Assicurati di lanciarlo con un launch file.")
            return
        conf = self.configure()
        
        if not conf:
            rospy.logfatal("Erorr in the K-Means configuration.")
            return
        else:
            rospy.loginfo("K-Means configurated correctly.")
        
        rospy.Subscriber('/cvsa/eeg_power', eeg_power, self.callback)
        self.pub = rospy.Publisher('/cvsa/neuroprediction/kmeans', NeuroOutput, queue_size=10)
        
        rospy.spin()
        
    def configure(self):
        rospy.loginfo(f"Loading K-Means from: {self.path_decoder}")
        with open(self.path_decoder, 'r') as file:
            params = yaml.safe_load(file)['KmeansModelCfg']['params']
            
        self.kmeans_name = yaml.safe_load(open(self.path_decoder, 'r'))['KmeansModelCfg']['name']
        
        # Load parameters
        self.mu = np.array(params['mu'])
        self.sigma = np.array(params['sigma'])
        self.centroids = np.array(params['centroids'])
        self.K = int(params['K'])
        
        # Controllo di sanità
        if self.K != self.centroids.shape[0]:
            rospy.logwarn("K in the YAML is different from the n. centroids!")
            return False
            
        self.o_l     = np.sort(np.array(params['occipital_left_idx']) - 1)
        self.o_r     = np.sort(np.array(params['occipital_right_idx']) - 1)
        self.frontal = np.sort(np.array(params['frontal_idx']) - 1)
        self.c_l     = np.sort(np.array(params['central_left_idx']) - 1)
        self.c_r     = np.sort(np.array(params['central_right_idx']) - 1)
        self.exclude_chs     = np.sort(np.array(params['excluded_idx']) - 1)
        
        self.nfeatures = int(params['nfeatures'])
        
        self.bands_features = np.array(params['band'])
        
        return True


    def compute_sparsity_features(self, signal, nbands):
        sparsity = np.zeros(self.nfeatures)
        
        for idx_band in range(0, nbands):
            c_signal = signal[idx_band,:]
            # --- 1. Feature LI (Lateralization Index) ---
            P_left_window = np.mean(c_signal[self.o_l])
            P_right_window = np.mean(c_signal[self.o_r])

            denominator = P_right_window + P_left_window + np.finfo(float).eps
            LAP_history = (P_right_window - P_left_window) / denominator
            sparsity[0] = np.abs(LAP_history) # LI

            # --- 2. Feature GI (Gini * Occipital Power) ---
            all_chs = np.arange(len(c_signal))
            non_eog_chs = np.setdiff1d(all_chs, self.exclude_chs)
            global_mean = np.mean(c_signal[non_eog_chs])
            current_signal_normalized = c_signal - global_mean
            mean_roi = np.array([
                np.mean(current_signal_normalized[self.frontal]),
                np.mean(current_signal_normalized[self.c_l]),
                np.mean(current_signal_normalized[self.c_r]),
                np.mean(current_signal_normalized[self.o_l]),
                np.mean(current_signal_normalized[self.o_r])
            ])
            mean_roi = np.abs(mean_roi)
            mean_roi_ordered = np.sort(mean_roi)
            n = len(mean_roi_ordered)
            total_sum = np.sum(mean_roi_ordered)
            if total_sum > 0:
                sum_roi_p = 0
                for i in range(n): 
                    sum_roi_p += (n - i) * mean_roi_ordered[i]

                gi = (1.0 / n) * (n + 1.0 - (2.0 * sum_roi_p) / total_sum)
            else:
                gi = 0

            pot_occipital = mean_roi[3] + mean_roi[4] 
            pot_total_roi = np.sum(mean_roi)

            if pot_total_roi > 0:
                occipital_power = pot_occipital / pot_total_roi
            else:
                occipital_power = 0

            sparsity[1] = occipital_power * gi # GI


            # --- 3. Feature GB (Global Brain Activity) ---
            sparsity[2] = global_mean # GB
        
        return sparsity

    def classify(self, dfet):
        if dfet is None:
            return 
        
        dfet_std = (dfet - self.mu) / self.sigma
        dfet_std_2d = dfet_std.reshape(1, -1)
  
        distances = cdist(dfet_std_2d, self.centroids, 'euclidean')
        neg_distances = -distances[0] 
        
        exp_values = np.exp(neg_distances - np.max(neg_distances))
        probabilities = exp_values / np.sum(exp_values)
        
        return probabilities
        
    def callback(self, msg):
        data = msg.data
        nchannels = msg.nchannels
        nbands = msg.nbands
        all_bands = np.array(msg.bands).reshape(-1, 2)
        
        reshaped_data = np.array(data).reshape(nchannels, nbands)
        
        tmp = []
        for i, c_band_features in enumerate(self.bands_features):
            for j, filter_band in enumerate(all_bands):
                if np.array_equal(c_band_features, filter_band):
                    tmp.append(reshaped_data[:, j])
                    break 
 
        dfet = self.compute_sparsity_features(np.array(tmp), nbands)
        
        probabilities = self.classify(dfet)
        
        hard_pred_vector = np.zeros(self.K, dtype=int)
        hard_pred_vector[np.argmax(probabilities)] = 1
        
        # publish the output
        output = NeuroOutput()
        output.header.stamp = rospy.Time.now()
        output.neuroheader.seq = msg.seq
        output.softpredict.data = probabilities.tolist()
        output.hardpredict.data = hard_pred_vector.tolist() 
        output.decoder.type = self.kmeans_name
        output.decoder.path = self.path_decoder
        
        self.pub.publish(output)
        
   
if __name__ == '__main__':
    KMeansClassifier()