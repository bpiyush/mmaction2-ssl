
### Download and setup dataset

1. Download & extract dataset
```bash
pip install gdown
cd ../../../data/ucf101_24/
gdown https://drive.google.com/uc?id=1PqKKW_kz00qT_fcheLf9LdWyy7tJr0o_
tar -zvxf UCF101_v2.tar.gz --directory ./
```

2. Download annotations from [here](https://github.com/gurkirt/corrected-UCF101-Annots): Download `pyannot.pkl` from here.

The dataset folder structure should look like:
```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf101_24
│   |   ├── brox-images
│   |   |   ├── Basketball
│   |   |   |   ├── v_Basketball_g01_c01
│   |   |   |   |   ├── 00001.jpg
│   |   |   |   |   ├── 00002.jpg
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00140.jpg
│   |   |   |   |   ├── 00141.jpg
│   |   |   ├── ...
│   |   |   ├── WalkingWithDog
│   |   |   |   ├── v_WalkingWithDog_g01_c01
│   |   |   |   ├── ...
│   |   |   |   ├── v_WalkingWithDog_g25_c04
│   |   ├── rgb-images
│   |   |   ├── Basketball
│   |   |   |   ├── v_Basketball_g01_c01
│   |   |   |   |   ├── 00001.jpg
│   |   |   |   |   ├── 00002.jpg
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00140.jpg
│   |   |   |   |   ├── 00141.jpg
│   |   |   ├── ...
│   |   |   ├── WalkingWithDog
│   |   |   |   ├── v_WalkingWithDog_g01_c01
│   |   |   |   ├── ...
│   |   |   |   ├── v_WalkingWithDog_g25_c04
│   |   ├── UCF101v2-GT.pkl
│   |   ├── pyannot.pkl
```
