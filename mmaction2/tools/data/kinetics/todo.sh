#!/bin/sh

DATASET="kinetics400"
taskset -a -c 0-24 ./download_annotations.sh $DATASET
taskset -a -c 0-24 ./download_videos.sh $DATASET
taskset -a -c 0-24 ./rename_classnames.sh $DATASET
taskset -a -c 0-24 ./generate_videos_filelist.sh $DATASET

