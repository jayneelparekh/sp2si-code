import numpy as np
import librosa
import librosa.core as core

import os, sys
sys.path.append(os.getcwd() + '/VocalMelodyExtPatchCNN-master')
import MelodyExt

sr = 16000
nfft = 1024
wlen = 1024
hop = 128

suffix_dict = {'B1':'vsp2si_nomelody', 'B2':'vsp2sibase', 'AllNorm':'vsp2si8.4nd', 'PMSE':'vsp2si8.3nd', 'PMTL':'vsp2si8.8sm_phnencE8', 'PhSync':'vsp2si8.3', 'AE':'vsp2siae'}
suffix_dict['b1'] = suffix_dict['B1']
suffix_dict['b2'] = suffix_dict['B2']
suffix_dict['ae'] = suffix_dict['AE']
suffix_dict['pmse'] = suffix_dict['PMSE']
suffix_dict['pmtl'] = suffix_dict['PMTL']
suffix_dict['allnorm'] = suffix_dict['AllNorm']
suffix_dict['phsync'] = suffix_dict['PhSync']


cmu_phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh', 'sp', 'il']
phn_dict = {}
for i in range(len(cmu_phn)):
    vec = np.zeros(len(cmu_phn))
    vec[i] = 1
    phn_dict[cmu_phn[i]] = i + 1


fld = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW', 'ZHIY']
psongs = [['01','09',13,18], ['05','08',11,15], ['07',15,16,20], ['04',10,12,17], ['04',10,12,17], ['05',11,19,20], ['02','03','06',14], ['07',15,16,20], ['05','08',11,15], ['01','09',13,18], ['05',11,19,20], ['02','03','06',14]]
		
all_pitch_contour = {}
for i in range(len(fld)):
    cur_fld = fld[i]
    for snum in psongs[i]:
        all_pitch_contour[cur_fld + '_' + str(snum)] = np.load('melody_contour/' + cur_fld + '_' + str(snum) + '.npy')



def gl_rec(mag_stft, hop, wlen, init_rec, n_iter=40):
    # Function for Griffin-Lim reconstruction
    rec = 1.0*init_rec
    rec_stft = core.stft(rec, n_fft=nfft, hop_length=hop, win_length=wlen)
    angles = rec_stft/np.abs(rec_stft)
    for i in range(n_iter):
        rec = core.istft(np.abs(mag_stft**1.2) * angles, hop, wlen)
        rec_stft = core.stft(rec, n_fft=nfft, hop_length=hop, win_length=wlen)
        angles = rec_stft/np.abs(rec_stft)
    return rec


def fastgl_rec(mag_stft, hop, wlen, n_iter=40):
    angles = np.exp(2j * np.pi * np.random.rand(*mag_stft.shape))
    momentum = 1.1
    rebuilt = 0

    for i in range(n_iter):
        tprev = 1*rebuilt
        inverse = core.istft(np.abs(mag_stft**1.2) * angles, hop_length=hop, win_length=wlen)      
        rebuilt = core.stft(inverse, n_fft=nfft, hop_length=hop, win_length=wlen)
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + 1e-16

    return inverse



def comp_lsd(ref_file, pred_file):
    ref = core.load(ref_file, sr=sr)[0]
    pred = core.load(pred_file, sr=sr)[0]
    stft_ref = np.abs(core.stft(ref, n_fft=nfft, hop_length=hop, win_length=wlen))
    stft_pred = np.abs(core.stft(pred, n_fft=nfft, hop_length=hop, win_length=wlen))
    logstft_ref = np.log(0.1 + stft_ref)
    logstft_pred = np.log(0.1 + stft_pred[:, :stft_ref.shape[1]])
    lsd = np.mean( np.sqrt(   np.sum((logstft_ref[7:220] - logstft_pred[7:220])**2, axis=0)   ) )
    return lsd


