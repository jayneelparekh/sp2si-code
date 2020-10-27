# Speech-to-Singing Conversion

* This repo is for the paper:  ["Speech-to-Singing Conversion in an Encoder-Decoder Framework"](https://arxiv.org/abs/2002.06595) by *Jayneel Parekh, Preeti Rao, Yi-Hsuan Yang* in ICASSP 2020.

* Link for the [project webpage](https://jayneelparekh.github.io/icassp20/)

# Setup
You can setup a new conda environment with the ```environment.yml``` file (recommended). You can start with [miniconda installation](https://docs.conda.io/en/latest/miniconda.html) if you are completely unfamiliar with anaconda   
   ```sh
   conda env create -f environment.yml
   ```
You will need to download the dataset (NUS-48E), models weigths, and clone a repo for melody extraction to run this code. 

* The NUS-48E dataset can be downloaded from <a href="https://smcnus.comp.nus.edu.sg/nus-48e-sung-and-spoken-lyrics-corpus/" rel="nofollow"> this link</a>. The downloaded dataset (folder named 'NUS_48E') should be saved inside this repository (current working directory). 

* The model weights can be downloaded from <a href="https://drive.google.com/file/d/18IiV4c-OBw2gnldlo9s7z8_Bzy6iKD0H/view?usp=sharing" rel="nofollow"> here</a>. You should place the complete folder (named 'models') in the 'output' folder of this repo.

* Download code for melody extractor system (by Li Su) [here](https://github.com/leo-so/VocalMelodyExtPatchCNN). You should extract the zip and place the folder named 'VocalMelodyExtPatchCNN-master' inside this repo (current working directory). After that you should move the file 'VocalMelodyExtPatchCNN-master/model3_patch25' with the other files of the repo (in the current working directory) 


# Usage
The first time you run the code, it will also organize the audio from NUS-48E in a dictionary. This can take up to 10-15 minutes.

You can currently 
1. Compute LSD for different models on random samples generated from NUS-48E dataset (with the function eval_sys()).
2. Compute random predictions for multiple models on the NUS-48E data (function random_pred()).
3. Compute prediction of a model on any given speech, melody file (function eval())

Both functions eval_sys() and random_pred() have some common set of arguments:
* *model_list*: Specifies the list of models you want to compute results for. Eg. \['PMTL', 'PMSE'\]. Current available model options are 'PMTL', 'PMSE', 'B1'. 'B2'.
* *n_samp*: Number of samples for prediction. 
* *min_length*: Minimum length of the input speech signal (in seconds). Default value is 1.0
* *fld*: List of singer folders from NUS_48E dataset for whom you want to conduct evaluation. Eg. \['ADIZ'\]. Default value is \['ADIZ', 'SAMF'\].
* *psongs*: List of songs for evaluation, for each singer specified in fld. Eg. \[\['01', '18'\]\]. Default value is \[\['18'\], \['18'\]\].
* *random* (Not available with random_pred() function): Denotes if input samples used for prediction are randomly selected from the generated samples. Default value is True

Arguments for function eval()
* *net1, net2*: Networks which you want to use for prediction
* *speech_file_loc*: Location of file containing speech
* *melody_file_loc*: Location of file to extract melody from

```
python evaluation_sp2si.py
```
