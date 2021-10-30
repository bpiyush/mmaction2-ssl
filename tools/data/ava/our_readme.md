0. Create symlink 
```bash
cd /go/to/repo/
mkdir data/
ln -s /ssd/pbagad/datasets/* data/
```

1. Download annotations
```bash
cd tools/data/ava/
VERSION=2.1 bash download_annotations.sh
VERSION=2.2 bash download_annotations.sh
```

2. Download videos
```bash
bash download_videos.sh
```
Convert `webm` videos to `mp4`.
```bash
python webm_to_mp4.py -o /ssd/pbagad/datasets/ava/
```

3. Cut videos
```bash
bash cut_videos.sh
```

4. Extract RGB frames
```bash
bash extract_rgb_frames_ffmpeg.sh
```