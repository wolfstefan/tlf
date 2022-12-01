# Transformer-based Late Fusion for Fine-Grained Object Recognition in Videos

This is the repository belonging to the RWS @ WACV2023 paper *A Transformer-based Late-Fusion Mechanism for Fine-Grained Object Recognition in Videos*. The following paragraphs will explain how to get up and running with the same setup used to derive the results in the paper.

## Prerequisites

The [MMAction2 framework](https://github.com/open-mmlab/mmaction2) is used to run our experiments. Accordingly, the mmaction2 directory contains a modified version of the upstream repository, forked at this commit hash: *88ffae3a5099d7eb3a4c8c21cfbdaec9e03ca7b7*.

Our repo also contains the code found in the [Video Swin Transformer repository](https://github.com/SwinTransformer/Video-Swin-Transformer) at hash: *db018fb8896251711791386bbd2127562fd8d6a6*.

**We can not guarantee that merging newer versions of these repositories into the provided state of mmaction2 won't break the code.** To discourage this further, we do not provide our git repository of mmaction2, but just the raw data.

As for prerequisites, we do not require much setup [in addition to what is covered by MMAction2](https://mmaction2.readthedocs.io/en/latest/install.html). Specifically, a recent version of Python (we use Python 3.7.3) and Decord is necessary. To fetch the YouTube-Cars and YouTube-Birds datasets, FFMPEG and YT-DLP need to be installed.

**Make sure to run through the [MMAction2 setup](https://mmaction2.readthedocs.io/en/latest/install.html) using our mmaction2 directory before continuing!**

## Fetching the data

The [YouTube-Cars and YouTube-Birds datasets](https://zhuchen03.github.io/projects/fgvc/) can be put anywhere, but need to be [symlinked](https://man7.org/linux/man-pages/man1/ln.1.html) to the *data* directory **inside mmaction2**. This directory might not exist already, in which case it needs to be created manually. The config files expect the respective directories to be called *data/YouTube-Cars* and *data/YouTube-Birds*. If the names in the data directory do not match the config files, the experiments will crash.

Once the *mmaction2/data* directory contains two links to the YouTube-Cars and YouTube-Birds folders, extract the JSON files provided by the dataset maintainer into them. The relevant files are *label2name.json*, *train_list.json* and *test_list.json*.

Fetching the actual video data is achieved through the scripts provided by us. Next to the mmaction2 directory in the root of the repository, a directory called *scripts* contains multiple files you can execute using Python:

1. *fetch_cars.py* - fetches the YouTube-Cars dataset into data/YouTube-Cars
2. *fetch_birds.py* - fetches the YouTube-Birds dataset into data/YouTube-Birds
3. *generate_file_list.py* - generates train_index.txt and test_index.txt, linking each video file to a ground truth label

Both fetch scripts perform multiple operations: they fetch each video from YouTube, remove the audio data to save disk space and reencode the video file to H264 MP4. While different codecs like VP9 should work in theory, our experiments failed to decode non-H264 video files.

**tl;dr: To get the data, create the following directories: mmaction2/data/YouTube-Cars and mmaction2/data/YouTube-Birds. Then, run at least one of the fetch scripts. Finally, run generate_file_list.py to build the ground truth indices.**

## Running experiments

Training and testing models is achieved simply by using the [train/test scripts provided by MMAction2](https://mmaction2.readthedocs.io/en/latest/getting_started.html#train-a-model). All config files used to derive are results can be found inside *mmaction2/configs/recognition/*.