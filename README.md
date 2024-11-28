# Code for submission 1942

## Dataset
- `CIRA-CIC-DoHBrw-2020`: https://www.unb.ca/cic/datasets/dohbrw-2020.html
- `Malicious_TLS`: https://github.com/gcx-Yuan/Malicious_TLS
  
## Data Preprocessing
- `preprocess_frame_len_DOH20.py`
 - Extracts packet length sequences from raw network traffic
 - Processes flows using 5-tuple info (src/dst IP, src/dst port, protocol)
 - Generates fixed-length (120) sequences with zero padding
 - Outputs preprocessed data in a format ready for model training

## Core Model Architecture
- `CNN_AE_updated.py`
 - Implements the traffic representation module with:
   - 1D CNN encoder/decoder backbone
   - Multi-head attention mechanism (4 attention heads) 
   - Batch normalization and ELU activation

## Support Modules
- `LNL.py`&`LNL_DOH.py`: Implementation of label noise learning components
- `model_MoCo.py`: 
  - Prototype updating mechanism
  - Contrastive learning components
  - Confidence sample selection

## Complete Experiment Scripts
- `main_DOH.ipynb` & `main_malicious_TLS.ipynb`
 - Full experiments on CIRA-CIC-DoHBrw-2020 dataset and Malicious TLS dataset
 - Integrates noise-resilient representation enhancement:
   - Latent Representation Preservation 
   - Cluster-driven Representation Differentiation 
   - Prototype-guided Representation Refinement
- Integrates unsupervised detection

As the paper is currently under review, the complete code will be released upon acceptance.
