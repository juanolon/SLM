# SLM: ShortList Model
Implementation of ShortList Model and its applications.


## Environment
```bash
conda env create -f slm.yaml
```

Since there're many dependences, we provide a quick-solving pathway building from **pytorch:2.1.0-cu11.8.0-py3.9**
```bash
# build slm base
pip install datasets einops fsspec git-lfs h5py hydra-core lightning nvitop omegaconf packaging pandas rich seaborn scikit-learn timm transformers triton
# check versions
# flash_attn:2.7.2.post1+cu11torch2.1cxx11abiFALSE-cp39   causal_conv1d:1.5.0.post8+cu11torch2.1cxx11abiFALSE-cp39   mamba_ssm:2.2.4+cu11torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install pika torchdiffeq
# build slm protein
pip install evodiff biopython
# build slm dna
pip install pyBigWig pytabix selene_sdk pyranges cooltools
```


### Preparation

#### Text8 Dataset
Download preprocessed text8 dataset or set the data_dir in file `./configs/data/text8.yaml`: cache_dir. If so, automatic download will be done.

#### Uniref50 Dataset
Download Uniref50 data as [Evodiff](https://zenodo.org/records/6564798) [Repo](https://github.com/microsoft/evodiff/issues/10) with such files:
```bash
---uniref50
  | consensus.fasta 
  | lengths_and_offsets.npz
  | splits.json
  | uniref50.tar.gz 
```
Then set uniref50 dir to file `./configs/data/uniref50.yaml`: cache_dir

#### DNA Enhancer Dataset
Download [dataset](https://zenodo.org/records/10184648) and organize files like that:
```bash
---dna_data
  | DeepMEL2_data.pkl
  | DeepFlyBrain_data.pkl
```

#### DNA Promoter Dataset
Download [dataset](https://zenodo.org/records/7943307) and organize files like that:
```bash
---promoter_design
  | ._agg.minus.bw.bedgraph.bw.gz
  | ._agg.plus.bw.bedgraph.bw.gz
  ...
  ...
  | Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai
  | Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap
```


#### GPT2 Model
GPT2 are only used for evaluation, download [gpt2-large](https://huggingface.co/openai-community/gpt2-large/tree/main) and report dir to `./configs/config.yaml`:eval.gen_ppl_eval_model_name_or_path


-----
## Training 
#### Text8 experiements
Run `make train_text8` to start training, sentences will be sampled eval 500 steps.

#### Protein experiments
Run `make train_uniref50` to start training, sequences will be sampled eval 500 steps.

#### DNA experiments
For DNA-Enhancer Design, please Run `make train_fb` and `make train_mel` to train SLM with different dataset.

For DNA-Promoter Design, run `make train_promoter` instead.



## Load from checkpoints and Sampling

We've released our [SLM model](https://huggingface.co/GenSI/SLM) on GenSI's huggingface, download ckpt files and replace the CKPT_PATH in makefile targets.


Run `make sample_uniref50` to sample protein sequences.


Run `make sample_fb` and `make sample_mel` to sample dna enhancer sequences.
