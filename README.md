# Vision Transformers for Video Dehazing : Ministry of Oceans and Fisheries
[Vision Transformers for Single Image Dehazing](https://github.com/IDKiro/DehazeFormer)


### Requirements
- Nvidia Driver Version 525 >
- Docker Version 20.10 >

### How to use
```
$ git clone https://github.com/Monologue2/dehazer.git
$ cd dehazer
$ docker compose up -d
$ docker exec -it dehazer python dehaze.py -i ./{FilePath} -o {Filename}
```
- Output file path
```
.
└── **output**
    └── **OutputFile.mp4**
```

### Test

```Bash
$ docker exec -it dehazer python dehaze.py -i ./files/test.mp4 -o test.mp4
==> Extracting Frames
100%|███████████████████████████████████████████████████████████████| 1749/1749 [00:43<00:00, 40.63it/s]
==> Start Dehazing...
100%|███████████████████████████████████████████████████████████████| 1749/1749 [04:34<00:00,  6.36it/s]
==> Merging...
100%|███████████████████████████████████████████████████████████████| 1749/1749 [00:45<00:00, 38.07it/s]
```
- Output
```
.
├── datasets
│   ├── __init__.py
│   └── loader.py
├── dehaze.py
├── docker-compose.yml
├── Dockerfile
├── **files**
│   └── **test.mp4**
├── models
│   ├── dehazeformer.py
│   ├── __init__.py
├── **output**
│   └── **test.mp4**
├── README.md
├── requirements.txt
├── saved_models
│   └── dehazeformer.pth
└── utils
    ├── common.py
    ├── data_parallel.py
    ├── __init__.py
    └── video.py
```


