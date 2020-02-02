# Speech-to-Singing Conversion

* This repo is for the paper:  "Speech-to-Singing Conversion in an Encoder-Decoder Framework" by *Jayneel Parekh, Preeti Rao, Yi-Hsuan Yang* in ICASSP 2020.

* Link for the [project webpage](https://jayneelparekh.github.io/icassp20/)

* Documentation is being updated currently

# Setup
You can setup a new conda environment with the ```environment.yml``` file (recommended). You can also start with [miniconda installation](https://docs.conda.io/en/latest/miniconda.html) if you are completely unfamiliar with anaconda   
   ```sh
   conda env create -f environment.yml
   ```

# Data
You will need to download the dataset (NUS-48E) and the models weigths to run the code. 

* The NUS-48E dataset can be downloaded from <a href="https://smcnus.comp.nus.edu.sg/nus-48e-sung-and-spoken-lyrics-corpus/" rel="nofollow"> this link</a>. The downloaded dataset (folder named 'NUS_48E') should be placed in this repo. 

* The model weights can be downloaded from <a href="https://drive.google.com/file/d/18IiV4c-OBw2gnldlo9s7z8_Bzy6iKD0H/view?usp=sharing" rel="nofollow"> here</a>. You should place the complete folder (named 'models') in the 'output' folder of this repo.

# Usage (Being updated)
You can currently 
1. Compute LSD for different models on NUS-48E dataset
2. Compute random predictions for multiple models on the NUS-48E data.

```
python evaluation_sp2si.py

```

# To be added
1. Pitch extraction related stuff
2. General function to run the model on any speech file and singing file
Need to rewrite these parts as I lost them when my hard drive crashed. Will add soon (within a few weeks)
