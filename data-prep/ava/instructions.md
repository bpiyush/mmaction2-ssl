
## Setup for AVA Dataset

[Source](https://github.com/cvdfoundation/ava-dataset) | [Project page](https://research.google.com/ava/)


### Install instructions

```bash
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch -y
pip install git+https://github.com/open-mmlab/mim.git
mim install mmaction2
pip install ipdb ipython tqdm wget
conda install av -c conda-forge -y
pip install PyTurboJPEG
```

### Download dataset
If you want to download the dataset in a folder `/path/to/folder/`, then use the following command.

```bash
python download.py -o /path/to/folder/
```