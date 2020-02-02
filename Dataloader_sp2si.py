import os
import torch
import librosa
import librosa.core as core
import numpy as np
import random
import scipy

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

import itertools
import pickle

import utils

phn_dict = utils.phn_dict
all_pitch_contour = utils.all_pitch_contour


#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------


class NUS_48E(Dataset):

    def __init__(self, root_dir, stft_params):
        self.root_dir = root_dir # Directory of the dataset, not of the repo
        self.sr = stft_params[0]
        self.nfft = stft_params[1]
        self.wlen = stft_params[2]
        self.hop = stft_params[3]
	self.all_audio = {}
	#self.read_all_audio()
        try:
            self.load_all_audio()
        except:
            self.read_all_audio()

        self.fld = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW', 'ZHIY']

    
    def __len__(self):
        # It's not really being used in the code as dataset size also determined by sampler, so just a random value
        return (300*10)

    def read_all_audio(self):
        print ('Reading audio, to collect all of it in a dictionary')
	psongs = [['01','09',13,18], ['05','08',11,15], ['07',15,16,20], ['04',10,12,17], ['04',10,12,17], ['05',11,19,20], ['02','03','06',14], ['07',15,16,20], ['05','08',11,15], ['01','09',13,18], ['05',11,19,20], ['02','03','06',14]] # List of all singers and their songs

	for i in range(len(self.fld)):
	    usr = self.fld[i]
	    for snum in psongs[i]:
		file_path1 = self.root_dir + str(usr) + '/sing/' + str(snum) + '.wav'
		file_path2 = self.root_dir + str(usr) + '/read/' + str(snum) + '.wav'

		audio1 = core.load(file_path1, self.sr)[0]
		audio2 = core.load(file_path2, self.sr)[0]

		self.all_audio[file_path1] = audio1
		self.all_audio[file_path2] = audio2

        f = open('NUS_data_dict.pkl', 'wb') # Saved and read from the project directory
        pickle.dump(self.all_audio, f)
        f.close()
        print "All audio read & stored"

    def load_all_audio(self):
        pkl_file = open('NUS_data_dict.pkl', 'rb')
        self.all_audio = pickle.load(pkl_file)
        pkl_file.close()
        print "All audio read & stored"
	


    def __getitem__(self, samp_info):

        usr = samp_info[0]   # Which user
	snum = samp_info[1]   # Which song of the user
	inp_start = float(samp_info[4]) * self.sr  # Start index of the time-domian signal
	inp_end = float(samp_info[5]) * self.sr   # End index of the time-domain signal
        lines_read = samp_info[6]
        lines_sung = samp_info[7]
        
        inp_audio = np.array([])
        file_path = self.root_dir + str(usr) + '/read/' + str(snum) + '.wav'
	inp_audio = self.all_audio[file_path][int(inp_start):int(inp_end)]
        inp_audio = remove_silent_frames(inp_audio)

	rps = np.random.uniform(-1.0, 1.0)
	inp_rps = librosa.effects.pitch_shift(inp_audio, self.sr, n_steps=rps)
        stft_inp = core.stft(inp_audio, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen)
	stft_rps = core.stft(inp_rps, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen)
        
	out_start = float(samp_info[2]) * self.sr  # Start index of the time signal
	out_end = float(samp_info[3]) * self.sr   # End index of the time signal

        file_path = self.root_dir + str(usr) + '/sing/' + str(snum) + '.wav'
        #out_audio = core.load(file_path, self.sr)[0][int(out_start):int(out_end)]
	out_audio = self.all_audio[file_path][int(out_start):int(out_end)]
        out_rps = librosa.effects.pitch_shift(out_audio, self.sr, n_steps=rps)
        stft_out = core.stft(out_audio, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen)
        stft_rps_out = core.stft(out_rps, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen)

	rate = stft_inp.shape[1]*1.0/stft_out.shape[1]
	stft_inp_orig = 1*stft_inp
	stft_inp = core.phase_vocoder(stft_inp, rate, self.hop)
	stft_inp = stft_inp[:, :stft_out.shape[1]]
	stft_rps = core.phase_vocoder(stft_rps, rate, self.hop)
	stft_rps = stft_rps[:, :stft_out.shape[1]]

        #phn_matrix = np.zeros([len(cmu_phn), stft_out.shape[1]])
        phn_matrix = np.zeros(stft_out.shape[1]).astype(int)
        hop_time = (self.hop*1.0/self.sr)
        for idx in range(0, len(lines_read)):
            phn_start, phn_end = extract_time(lines_sung[idx])
            if (lines_sung[idx][-3] == ' '):
                cur_phn = lines_sung[idx][-2:-1] 
            elif (lines_sung[idx][-4] == ' '):
                cur_phn = lines_sung[idx][-3:-1]

            if (cur_phn[-1] == ' '):
                cur_phn = cur_phn[0]
            start_time = float(samp_info[2])
            end_time = float(samp_info[3])
            if (phn_end - phn_start > 0.005): # Just a check that the phone should sustain for more than a few ms
                start_idx = int((phn_start - start_time)/hop_time)
                end_idx = int((phn_end - start_time)/hop_time)
                #print start_idx, end_idx, cur_phn, phn_matrix.shape
                #phn_matrix[:, start_idx:end_idx] = np.tile(phn_dict[cur_phn], (end_idx-start_idx, 1)).transpose()        
                phn_matrix[start_idx:end_idx] = int(phn_dict[cur_phn])

        return [np.abs(stft_inp), np.abs(stft_out), pitch_max(np.abs(stft_inp)), pitch_pyin(int(out_start), usr, snum, stft_out.shape), rate, stft_inp_orig, stft_inp, stft_out, np.abs(stft_rps), pitch_max(np.abs(stft_rps)), np.abs(stft_rps_out), self.fld.index(usr), phn_matrix]
        #return [np.abs(stft_inp), np.abs(stft_rps_out), pitch_max(np.abs(stft_inp)), pitch_max(np.abs(stft_rps_out)), rate, stft_inp_orig, stft_inp, stft_rps_out, np.abs(stft_rps), pitch_max(np.abs(stft_rps)), np.abs(stft_out)]



class NUS_48E_dur(Dataset):

    def __init__(self, root_dir, stft_params):
        self.root_dir = root_dir
        self.sr = stft_params[0]
        self.nfft = stft_params[1]
        self.wlen = stft_params[2]
        self.hop = stft_params[3]
	self.all_audio = {}
	#self.read_all_audio()
        self.load_all_audio()
        self.fld = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW', 'ZHIY']

    
    def __len__(self):
        # Only approximate, not actual
        return (300*10)

    def read_all_audio(self):
	psongs = [['01','09',13,18], ['05','08',11,15], ['07',15,16,20], ['04',10,12,17], ['04',10,12,17], ['05',11,19,20], ['02','03','06',14], ['07',15,16,20], ['05','08',11,15], ['01','09',13,18], ['05',11,19,20], ['02','03','06',14]]

	for i in range(len(self.fld)):
	    usr = self.fld[i]
	    for snum in psongs[i]:
		file_path1 = self.root_dir + str(usr) + '/sing/' + str(snum) + '.wav'
		file_path2 = self.root_dir + str(usr) + '/read/' + str(snum) + '.wav'
	
		audio1 = core.load(file_path1, self.sr)[0]
		audio2 = core.load(file_path2, self.sr)[0]

		self.all_audio[file_path1] = audio1
		self.all_audio[file_path2] = audio2

	print "All audio read & stored"
        f = open('NUS_data_dict.pkl', 'wb')
        pickle.dump(self.all_audio, f)
        f.close()

    def load_all_audio(self):
        pkl_file = open('NUS_data_dict.pkl', 'rb')
        self.all_audio = pickle.load(pkl_file)
        pkl_file.close()
        print "All audio read & stored"
	


    def __getitem__(self, samp_info):

        #print samp_info[0], samp_info[1], samp_info[2], samp_info[3]
        usr = samp_info[0]   # Which user
	snum = samp_info[1]   # Which song out of user's 4 songs
	inp_start = float(samp_info[4]) * self.sr  # Start index of the time-domian signal
	inp_end = float(samp_info[5]) * self.sr   # End index of the time-domain signal
        lines_read = samp_info[6]
        lines_sung = samp_info[7]
        
        inp_audio = np.array([])
        file_path = self.root_dir + str(usr) + '/read/' + str(snum) + '.wav'
        #'''
        inp_full = self.all_audio[file_path]
        for idx in range(0, len(lines_read)):
            r_start, r_end = extract_time(lines_read[idx])
            s_start, s_end = extract_time(lines_sung[idx])
            stretch_rate = (r_end-r_start)/(1e-3+s_end-s_start)
            #print r_end, r_start, s_end, s_start
            inp_phn = inp_full[int(r_start * self.sr) : int(r_end * self.sr)]
            inp_phn_stretch = librosa.effects.time_stretch(inp_phn, stretch_rate)
            inp_audio = np.append(inp_audio, inp_phn_stretch)
        
	rps = np.random.uniform(-1.0, 1.0)
        #rps = 3.0
	inp_rps = librosa.effects.pitch_shift(inp_audio, self.sr, n_steps=rps)
        stft_inp = core.stft(inp_audio, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen)
	stft_rps = core.stft(inp_rps, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen)
        
	out_start = float(samp_info[2]) * self.sr  # Start index of the time signal
	out_end = float(samp_info[3]) * self.sr   # End index of the time signal

        file_path = self.root_dir + str(usr) + '/sing/' + str(snum) + '.wav'
        #out_audio = core.load(file_path, self.sr)[0][int(out_start):int(out_end)]
	out_audio = self.all_audio[file_path][int(out_start):int(out_end)]
        out_rps = librosa.effects.pitch_shift(out_audio, self.sr, n_steps=rps)
        stft_out = core.stft(out_audio, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen)
        stft_rps_out = core.stft(out_rps, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen)

	# Making input also length of 3 seconds (if not wanted comment next 2 lines)
	rate = stft_inp.shape[1]*1.0/stft_out.shape[1]
	stft_inp_orig = 1*stft_inp
	stft_inp = core.phase_vocoder(stft_inp, rate, self.hop)
	stft_inp = stft_inp[:, :stft_out.shape[1]]
	stft_rps = core.phase_vocoder(stft_rps, rate, self.hop)
	stft_rps = stft_rps[:, :stft_out.shape[1]]

        #phn_matrix = np.zeros([len(cmu_phn), stft_out.shape[1]])
        phn_matrix = np.zeros(stft_out.shape[1]).astype(int)
        hop_time = (hop*1.0/sr)
        for idx in range(0, len(lines_read)):
            phn_start, phn_end = extract_time(lines_sung[idx])
            if (lines_sung[idx][-3] == ' '):
                cur_phn = lines_sung[idx][-2:-1] 
            elif (lines_sung[idx][-4] == ' '):
                cur_phn = lines_sung[idx][-3:-1]

            if (cur_phn[-1] == ' '):
                cur_phn = cur_phn[0]
            start_time = float(samp_info[2])
            end_time = float(samp_info[3])
            if (phn_end - phn_start > 0.005):
                start_idx = int((phn_start - start_time)/hop_time)
                end_idx = int((phn_end - start_time)/hop_time)
                #print start_idx, end_idx, cur_phn, phn_matrix.shape
                #phn_matrix[:, start_idx:end_idx] = np.tile(phn_dict[cur_phn], (end_idx-start_idx, 1)).transpose()        
                phn_matrix[start_idx:end_idx] = int(phn_dict[cur_phn])


        return [np.abs(stft_inp), np.abs(stft_out), pitch_max(np.abs(stft_inp)), pitch_pyin(int(out_start), usr, snum, stft_out.shape), rate, stft_inp_orig, stft_inp, stft_out, np.abs(stft_rps), pitch_max(np.abs(stft_rps)), np.abs(stft_rps_out), self.fld.index(usr), phn_matrix]  # Input, Output, Original Input-Output length ratio
        #return [np.abs(stft_inp), np.abs(stft_rps_out), pitch_max(np.abs(stft_inp)), pitch_max(np.abs(stft_rps_out)), rate, stft_inp_orig, stft_inp, stft_rps_out, np.abs(stft_rps), pitch_max(np.abs(stft_rps)), np.abs(stft_out)]







class nus_samp(Sampler):
    def __init__(self, root_dir, n_batch, n_iter, fld, psongs, use_word=False, randomize=True, print_elem=False, min_len=0.0):
	self.root = root_dir
        self.batch_size = n_batch
	self.n_iter = n_iter
	self.fld = fld
	self.psongs = psongs
	self.all_samp_details = []
	self.segment_time = 3
	self.use_word = use_word
	self.randomize = randomize
        self.print_elem = print_elem
	self.cur_batch_idx = 0
	self.samp_lenwise = {}
        self.min_splen = min_len
	self.build_samp_inline()

	# Some stats variables
	self.inp_length = 0
	self.out_length = 0
	self.inout_ratio = 0

    def stats_segments(self):
	all_samp = self.all_samp_details
	self.inp_length = np.array([all_samp[k][3] - all_samp[k][2] for k in range(len(all_samp))])
	self.out_length = np.array([all_samp[k][5] - all_samp[k][4] for k in range(len(all_samp))])
	self.inout_ratio = self.inp_length / self.out_length
        return (self.inp_length, self.out_length, self.inout_ratio)
	



    def build_samp_inline(self):
	for i in range(len(self.fld)):
            cur_fld = self.fld[i]

            for snum in self.psongs[i]:
		phn_sung = open(self.root + 'nus-smc-corpus_48/' + cur_fld + '/sing/' + str(snum) + '.txt', "r")
                phn_read = open(self.root + 'nus-smc-corpus_48/' + cur_fld + '/read/' + str(snum) + '.txt', "r")

		line_start = 1
                line_end = 1
                all_lines_read = phn_read.readlines()
                all_lines_sung = phn_sung.readlines()

		# First mark all the lines with big pauses (in speech lines they are most probably silences, 'sil' with > 30ms)
		pause_line = [m for m in range(len(all_lines_sung)) if all_lines_sung[m][-4:-1] == 'sil']
		for k in range(len(pause_line)-1):
		    start_line = pause_line[k]
		    end_line = pause_line[k+1]
		    # start_line, end_line mark the lines within which all the samples will be selected
		    # Mark all the in between words now
		    word_line = [m for m in range(start_line, end_line+1) if all_lines_read[m][-4:-1] == 'sil']
		    #print len(word_line)
		    # now select two word sets
		    for seq_len in range(3, len(word_line)):
			cur_len_samples = []
		        for t in range(len(word_line)-seq_len):
			    line_start = word_line[t] + 1
			    line_end = word_line[t+seq_len] - 1
			    start = float(all_lines_read[line_start][0:8]) - 0.01
			    try:
			        end = float(all_lines_read[line_end][9:17]) + 0.01
			    except:
			        end = float(all_lines_read[line_end][11:19]) + 0.01

			    # Now select the input
			    out_start = float(all_lines_sung[line_start][0:8]) - 0.01
			    try:
			        out_end = float(all_lines_sung[line_end][9:17]) + 0.01
			    except:
			        out_end = float(all_lines_sung[line_end][11:19]) + 0.01
                            if (end - start > self.min_splen):
			        self.all_samp_details.append([cur_fld, snum, out_start, out_end, start, end, np.array(all_lines_read[line_start:line_end+1]), np.array(all_lines_sung[line_start:line_end+1]) ])
			    #self.samp_lenwise[seq_len].append([cur_fld, snum, inp_start, inp_end, start, end, line_start, line_end])



    def RandomSelect(self):
	if (self.use_word == False or self.use_word == True):
	    idx_arr = range(len(self.all_samp_details))
            #idx_arr = [1761, 74]
	    if (self.randomize):
	        random.shuffle(idx_arr)
	        idx_arr = idx_arr[:self.batch_size]
                if self.print_elem:
                    print "Samples retrieved: ", idx_arr
	    else:
		cur_batch = self.cur_batch_idx
		start_idx = cur_batch * self.batch_size
		end_idx = min(start_idx + self.batch_size, len(self.all_samp_details))
		idx_arr = range(start_idx, end_idx)
		print idx_arr
		self.cur_batch_idx += 1
		if(end_idx == len(self.all_samp_details)):
		    self.cur_batch_idx = 0
                #idx_arr = [527]
                #idx_arr = [93]
                #idx_arr = [464] # 93, 464, 615, 1352
                #idx_arr = [649]
                idx_arr = [1327]

	    #return np.array(self.all_samp_details)[idx_arr]
            #print idx_arr
            return [self.all_samp_details[idx_arr[0]]]
		    
	

    def __iter__(self):
	samples = []
	for i in range(self.n_iter):
            samples.append(self.RandomSelect())
        return iter(samples)

    def __len__(self):
        return self.n_iter * self.batch_size		    




def pitch_pyin(start_samp_num, usr, snum, out_shape, hop_size=128, fs=16000, nfft=1024):
    # This does not calculate, just retrieves the target pitch already stored
    # start_idx is the sample number
    full_pc = all_pitch_contour[usr + '_' + str(snum)]
    pc_start = max(0, start_samp_num)
    pc_start = int(pc_start/hop_size)
    pc_end = pc_start + out_shape[1]
    extract_pc = full_pc[pc_start:pc_end]
    # Now make the pitch contour image
    pc = np.zeros(out_shape)
    idx1 = (1.0 * extract_pc * (out_shape[0]-1)*2/fs).astype(int)
    idx2 = np.array(range(pc.shape[1]))
    pc[idx1, idx2] = 1
    pc[-1] = 1 * pc[0]
    pc[0] = 0 * pc[0]
    return pc


def pitch_max(inp):
    # pc is pitch contour
    # Assume inp to be magnitude spectrogram
    pc = np.zeros(inp.shape)
    idx1 = np.abs(np.argmax(inp, axis=0))
    idx2 = np.array(range(pc.shape[1]))
    pc[idx1, idx2] = 1
    return pc


def remove_silent_frames(x, dyn_range=40, framelen=1024, hop=128):
    # Compute Mask
    w = scipy.hanning(framelen + 2)[1:-1]
    x_frames = np.array(
        [w * x[i:i + framelen] for i in range(0, len(x) - framelen, hop)])
    # Compute energies in dB
    x_energies = 20 * np.log10(np.linalg.norm(x_frames, axis=1) + 1e-9)
    # Find boolean mask of energies lower than dynamic_range dB with respect to maximum clean speech energy frame
    mask = (np.max(x_energies) - dyn_range - x_energies) < 0
    # Remove silent frames by masking
    x_frames = x_frames[mask]
    n_sil = (len(x_frames) - 1) * hop + framelen
    x_sil = np.zeros(n_sil)
    for i in range(x_frames.shape[0]):
        x_sil[range(i * hop, i * hop + framelen)] += x_frames[i, :]
    return x_sil


def extract_time(line):
    start = float(line[0:8])
    try:
        end = float(line[9:17])
    except:
        end = float(line[11:19])
    return start, end









