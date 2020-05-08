import utils
import torch
import numpy as np
import librosa
import librosa.core as core
import os, sys
#import nus_model as Models
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import scipy
import networks as defModel
import Dataloader_sp2si as DL
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import random
import itertools

print "Imported all modules"

# Variables user can modify
#fld = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW', 'ZHIY']
#psongs = [['01','09',13], ['05','08',11,15], ['07',15,16,20], ['04',10,12,17], ['04',10,12,17], ['05',11,19,20], ['02','03','06',14], ['07',15,16,'20'], ['05','08',11, '15'], ['01','09',13], ['05',11,19,20], ['02','03','06',14]] # Training songs for each singer folder
test_fld = ['ADIZ', 'SAMF']
test_psongs = [['18'], ['18']]
model_list = ['PMTL', 'PMSE', 'B1', 'B2']
n_samp = 10 # Number of test samples
use_cuda = False

device = torch.device("cuda" if use_cuda else "cpu")

# Fixed global variables
sr, nfft, wlen, hop = 16000, 1024, 1024, 128 # Now fixed
augment = False  # Whether to collate the data augmentation part or leave it
data_key = os.getcwd() + '/NUS_48E/nus-smc-corpus_48/'#'/media/asr-gpu/hdd/User_data/jayneel/Style_Transfer/NUS_48E/nus-smc-corpus_48/'
data_dir = os.getcwd() + '/NUS_48E/'#'/ldaphome/jparekh/Style_Transfer/NUS_48E/'
suffix_dict = utils.suffix_dict


def my_collate_e8(batch):
    # What does input to this function, 'batch' contain?
    # Considering Dataloader_sp2si.py's nus_samp getitm it contains:
    # batch[i][0]: mag STFT of input - stretched
    # batch[i][1]: mag STFT of target
    # batch[i][2]: pitch "image" of input STFT (useless now)
    # batch[i][3]: pitch "image" of target STFT
    # batch[i][4]: rate of stretching for this example
    # batch[i][5]: Original input STFT, [6]: Stretched input STFT, [7]: Target STFT
    # batch[i][8]: mag STFT of pitch shifted input - stretched
    # batch[i][9]: pitch "image" of pitch shifted output STFT
    # batch[i][10]: mag STFT of pitch shifted target
    # batch[i][11]: User representation as one-hot coding

    # What does this function return? A list with indices containing the following:
    # 0: input tensor consisting of stretched speech input mag STFT's (padded with zeros)
    # 1: target tensor of singing (padded)
    # 2: rates of stretching in a numpy array. Not really used anywhere
    # 3: tensor with info about target melody contour
    # 4: Original input as numpy array (non-stretched version)
    # 5: Stretched input speech as numpy array 
    # 6: Target singing as numpy array
    # 7: Tensor consisting of pitch shifted stretched speech mag STFT
    # 8: Tensor of user ids (Not used anywhere)
    # 9: Frame-level phoneme information for singing

    # Will move this function to utils.py in the future 
 
    batch_inp, batch_out, batch_pitch_inp, batch_pitch_out, batch_rate, batch_inp_orig, batch_inp_phase, batch_out_phase, batch_out_ps, batch_usr, batch_phn = [], [], [], [], [], [], [], [], [], [], []
    size_inp = 0
    size_out = 0
    for i in range(len(batch)):
        size_inp = max(size_inp, batch[i][0].shape[1])
        size_out = max(size_out, batch[i][1].shape[1])

    size_inp += 8 - size_inp%8
    size_out += 8 - size_out%8  
    for i in range(len(batch)):
        # Concatenate with zeros to make no. of columns multiple of 8. Required for processing by the network.
        batch[i][0] = np.concatenate([batch[i][0], np.zeros([batch[i][0].shape[0], size_inp-batch[i][0].shape[1]])], axis=1)
        batch[i][1] = np.concatenate([batch[i][1], np.zeros([batch[i][1].shape[0], size_out-batch[i][1].shape[1]])], axis=1)
        batch[i][2] = np.concatenate([batch[i][2], np.zeros([batch[i][2].shape[0], size_inp-batch[i][2].shape[1]])], axis=1)
        batch[i][3] = np.concatenate([batch[i][3], np.zeros([batch[i][3].shape[0], size_out-batch[i][3].shape[1]])], axis=1)
        batch[i][10] = np.concatenate([batch[i][10], np.zeros([batch[i][10].shape[0], size_out-batch[i][10].shape[1]])], axis=1)
        batch[i][12] = np.concatenate((batch[i][12], np.zeros(size_out-batch[i][12].shape[0])))

        batch_inp.append(  np.log(1+batch[i][0])  )
        batch_out.append(  np.log(1+batch[i][1])  )
        batch_pitch_inp.append( batch[i][2] )
        batch_pitch_out.append( batch[i][3] )
        batch_out_ps.append(  np.log(1+batch[i][10])   )
        batch_usr.append( batch[i][11] )
        batch_phn.append( batch[i][12] )

        if (augment):
            batch[i][8] = np.concatenate([batch[i][8], np.zeros([batch[i][8].shape[0], size_inp-batch[i][8].shape[1]])], axis=1)
            batch[i][9] = np.concatenate([batch[i][9], np.zeros([batch[i][9].shape[0], size_out-batch[i][9].shape[1]])], axis=1)
            batch_inp.append(  np.log(1+batch[i][8])  )
            batch_out.append(  np.log(1+batch[i][1])  )
            batch_pitch_inp.append( batch[i][2] )   # Assuming pitch input is meaningless here
            batch_pitch_out.append( batch[i][3] )
            batch_usr.append( batch[i][11] )
            batch_phn.append( batch[i][12] )

        batch_rate.append(batch[i][4])
        batch_inp_orig.append(batch[i][5])
        batch_inp_phase.append(batch[i][6])
        batch_out_phase.append(batch[i][7])

    return [torch.from_numpy(np.array(batch_inp)).float(), torch.from_numpy(np.array(batch_out)).float(), np.array(batch_rate), torch.from_numpy(np.array(batch_pitch_out)).float(), batch_inp_orig, batch_inp_phase, batch_out_phase, torch.from_numpy(np.array(batch_out_ps)).float(), torch.from_numpy(np.array(batch_usr)), torch.from_numpy(np.array(batch_phn).astype(int))]




def random_pred(model_list=['PMTL'], n_samp=2, min_length=1.0, fld=test_fld, psongs=test_psongs):
    # Predicts output of specified list of systems on some random samples from the dataset
    # It also takes as arguments the number of samples for evaluation, minimum length of each sample, 
 
    nus_train_data = DL.NUS_48E(data_key, [sr, nfft, wlen, hop])
    sampler = DL.nus_samp(data_dir, 1, n_samp, fld, psongs, use_word=True, randomize=True, print_elem=True, min_len=min_length)
    dataload = DataLoader(dataset=nus_train_data, batch_sampler=sampler, collate_fn=my_collate_e8)
    samp_idx = -1
    lsd = []
    for data in dataload:
        # Initialize, Load the networks and their weights properly taking into account the exceptions
        samp_idx += 1
        print 'Processing sample', samp_idx
        for idx in range(len(model_list)):
            cur_model = model_list[idx]
            suffix = suffix_dict[cur_model]
            network2 = defModel.exp_net(512, 512, freq=513).to(device)
            if cur_model == 'B2' or cur_model == 'b2':
                network1 = defModel.net_base(512, 512, freq=513).to(device)
            else:
                network1 = defModel.net_in_v2(512, 512, freq=513).to(device)

            if not (cur_model == 'B1' or cur_model == 'b1'):
                network2.load_state_dict(torch.load('output/models/net2_' + suffix + '.pt', map_location=device)) # Complete
            network1.load_state_dict(torch.load('output/models/net1_' + suffix + '.pt', map_location=device))
            network1, network2 = network1.eval(), network2.eval()

            # Make predictions
            encode2 = int(not cur_model == 'B1') * network2(Variable(data[3].to(device)))

            pred, encode1 = network1(Variable(data[0].to(device)), encode2)
            pred = pred.cpu().data.numpy()
            pred[pred < 0] = 0

            #Save log-STFTs of input, target and prediction
            saving_dir = 'output/random_predictions/'
            logstft_inp = data[0].numpy()
            logstft_out = data[1].numpy()
            logstft_pred = 1.0*pred
            np.save(saving_dir + 'inp_lgstft' + str(samp_idx), logstft_inp)
            np.save(saving_dir + 'out_lgstft' + str(samp_idx), logstft_out)
            np.save(saving_dir + 'pred_lgstft' + str(samp_idx), logstft_pred)

            # Get time domain signals
            stft_pred = np.zeros([513, pred.shape[2]])
            stft_pred[:pred.shape[1]] = np.exp(pred[0]) - 1

            time_pred = utils.gl_rec(stft_pred, hop, wlen, core.istft(stft_pred**1.0, hop, wlen))  
            time_inp_orig = core.istft(data[4][0], hop, wlen)
            time_inp_phase = core.istft(data[5][0], hop, wlen)
            time_target_phase = core.istft(data[6][0], hop, wlen)

            # Save predictions
            librosa.output.write_wav(saving_dir + 'original_speech_' + str(samp_idx) + '.wav', time_inp_orig, sr)
            librosa.output.write_wav(saving_dir + 'stretched_speech_' + str(samp_idx) + '.wav', time_inp_phase, sr)
            librosa.output.write_wav(saving_dir + 'true_singing_' + str(samp_idx) + '.wav', time_target_phase, sr)
            librosa.output.write_wav(saving_dir + 'predicted_singing_' + str(samp_idx) + cur_model + '.wav', time_pred, sr)

    return





def eval_sys(model_list=['PMTL', 'PMSE', 'B1', 'B2'], n_samp=30, min_length=1.0, random=True, fld=test_fld, psongs=test_psongs):
    # Currently evaluates the specified models on the NUS dataset for the given songs. Default songs comprise of our test set
    # It also takes as arguments the number of samples for evaluation (n_samp), minimum length of speech in each sample (min_length),  
    # Returns array of all computed LSD's and prints the mean LSD for each model

    nus_train_data = DL.NUS_48E(data_key, [sr, nfft, wlen, hop])
    sampler = DL.nus_samp(data_dir, 1, n_samp, fld, psongs, use_word=True, randomize=random, print_elem=False, min_len=min_length)
    dataload = DataLoader(dataset=nus_train_data, batch_sampler=sampler, collate_fn=my_collate_e8)
    samp_idx = -1
    lsd = []
    for data in dataload:
        # Initialize, Load the networks and their weights properly taking into account the exceptions
        samp_idx += 1
        print 'Processing sample ', samp_idx
        for idx in range(len(model_list)):
            cur_model = model_list[idx]
            suffix = suffix_dict[cur_model]
            network2 = defModel.exp_net(512, 512, freq=513).to(device)
            if cur_model == 'B2':
                network1 = defModel.net_base(512, 512, freq=513).to(device)
            else:
                network1 = defModel.net_in_v2(512, 512, freq=513).to(device)

            if not cur_model == 'B1':
                network2.load_state_dict(torch.load('output/models/net2_' + suffix + '.pt', map_location=device)) # Complete
            network1.load_state_dict(torch.load('output/models/net1_' + suffix + '.pt', map_location=device))
            network1, network2 = network1.eval(), network2.eval()

            # Make predictions
            encode2 = int(not cur_model == 'B1') * network2(Variable(data[3].to(device)))
            pred, encode1 = network1(Variable(data[0].to(device)), encode2)
            pred = pred.cpu().data.numpy()
            pred[pred < 0] = 0

            #Temporarily save log-STFTs of input target and prediction
            logstft_inp = data[0].numpy()
            logstft_out = data[1].numpy()
            logstft_pred = 1.0*pred
            np.save('runtime_folder/inp_stft', logstft_inp)
            np.save('runtime_folder/out_stft', logstft_out)
            np.save('runtime_folder/pred_stft', logstft_pred)

            # Get time domain signals
            stft_inp = np.zeros([513, pred.shape[2]])
            stft_pred = np.zeros([513, pred.shape[2]])
            stft_target = np.zeros([513, pred.shape[2]])

            stft_pred[:pred.shape[1]] = np.exp(pred[0]) - 1
            time_pred = utils.gl_rec(stft_pred, hop, wlen, core.istft(stft_pred**1.0, hop, wlen))  
            time_target_phase = core.istft(data[6][0], hop, wlen)

            # Save predictions in the runtime folder
            true_file = 'runtime_folder/runtime_true.wav'
            pred_file = 'runtime_folder/runtime_pred.wav'
            librosa.output.write_wav(true_file, time_target_phase, sr)
            librosa.output.write_wav(pred_file, time_pred, sr)
            calc_lsd = utils.comp_lsd(true_file, pred_file)
            #print cur_model, calc_lsd
            lsd.append(calc_lsd)

    # Print the results
    arr = np.zeros([len(model_list), n_samp])
    for i in range(len(model_list) * n_samp):
        arr[i % len(model_list), i//len(model_list)] = lsd[i]
    for i in range(len(model_list)):
        print model_list[i] + ' (mean LSD):', np.mean(arr[i])

    return lsd


def eval(net1, net2, speech_file_loc, melody_file_loc):
    # Evaluates the result of net1, net2 on a given speech file and melody file
    # speech_file_loc, melody_file_loc are strings that specify the location of the respective audio files 
    network1, network2 = net1.eval(), net2.eval()
    # Read input audio
    orig_speech = core.load(speech_file_loc, sr)[0]
    inp_speech = DL.remove_silent_frames(orig_speech)
    #inp_speech = 1.0 * orig_speech
    stft_inp = core.stft(inp_speech, n_fft=nfft, hop_length=hop, win_length=wlen)
    
    # Extract melody and create its image
    melody = utils.MelodyExt.melody_extraction(melody_file_loc, 'runtime_folder/ref_melody')[0]
    ref_pc = melody[:, 1]
    ref_time = melody[:, 0]
    const = hop * 1.0 / sr
    new_sampling_time = np.arange(const, ref_time[-1], const)
    interp_melody = np.interp(new_sampling_time, ref_time, ref_pc)
    n_frames = new_sampling_time.shape[0]
    idx1 = (1.0 * interp_melody * nfft / sr).astype(int)
    idx2 = np.array(range(n_frames))
    pc = np.zeros([1 + nfft/2, n_frames])
    pc[idx1, idx2] = 1
    pc[-1] = 1 * pc[0]
    pc[0] = 0 * pc[0]
  
    # Complete input preprocessing
    rate = stft_inp.shape[1] * 1.0 / n_frames
    stft_inp = core.phase_vocoder(stft_inp, rate, hop) # Stretch input speech to target length
    n_frames += 8 - n_frames%8
    # Append zeros to make it suitable for network
    stft_inp = np.concatenate([stft_inp, np.zeros([stft_inp.shape[0], n_frames-stft_inp.shape[1]])], axis=1)
    pc = np.concatenate([pc, np.zeros([pc.shape[0], n_frames-pc.shape[1]])], axis=1)
    stft_inp = np.log(1 + np.abs(stft_inp))
    stft_inp, pc = torch.from_numpy(stft_inp).float().unsqueeze(0), torch.from_numpy(pc).float().unsqueeze(0) # Make tensors

    # Extract output
    encode2 = network2(Variable(pc.to(device)))
    pred, encode1 = network1(Variable(stft_inp.to(device)), encode2)
    pred = pred[0].cpu().data.numpy()
    pred[pred < 0] = 0
    pred = np.exp(pred) - 1
    time_pred = 3.0 * utils.gl_rec(pred, hop, wlen, core.istft(pred, hop, wlen)) # Adding a multiplier to increase loudness
    return time_pred  


if __name__ == '__main__':
    args = sys.argv[1:]
    suffix = suffix_dict['PMTL'] # Get the suffix of PMTL model
    net1 = defModel.net_in_v2(512, 512, freq=513).to(device)
    net2 = defModel.exp_net(512, 512, freq=513).to(device)
    net2.load_state_dict(torch.load('output/models/net2_' + suffix + '.pt', map_location=device))
    net1.load_state_dict(torch.load('output/models/net1_' + suffix + '.pt', map_location=device))
    random_pred(['pmtl', 'b2'])
    stats = eval_sys()


