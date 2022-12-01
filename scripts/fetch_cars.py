#!/usr/bin/python3

import json
import os
import sys
import shutil
import subprocess

from yt_dlp import YoutubeDL
from tqdm import tqdm
from pathlib import Path

def fetch(foldername, idlist, label2name):
    os.chdir(foldername)
    for vidid, labelid in tqdm(idlist, desc="Fetching {}/".format(foldername)):
        label = str(label2name[str(labelid)])
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

        download_failed = False
        audiorm_failed = False

        os.makedirs(label, exist_ok=True)
        os.chdir(label)

        try:
            with YoutubeDL({'outtmpl': '%(id)s.%(ext)s', }) as ydl:
                ydl.download(['http://www.youtube.com/watch?v={}'.format(vidid)]) 

            full_filename = [x for x in os.listdir(os.getcwd()) if x.startswith(vidid)][0]
            vfname, _ = os.path.splitext(full_filename)
            targetf = "{}-nosound.mp4".format(vfname) 

            subprocess.call(["ffmpeg", "-i", full_filename, "-vcodec", "libx264", "-an", targetf], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            os.remove(full_filename)
            os.rename(targetf, "{}.mp4".format(vfname))
        except:
            pass

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        os.chdir("..")
    os.chdir("..")

def clean(targetpath):
    for root, dirs, _ in os.walk(targetpath):
        for d in dirs:
            for root2, _, files in os.walk(os.path.join(root, d)):
                for f in files:
                    vidfile = os.path.join(root2, f)
                    fname, fext = os.path.splitext(vidfile)

                    if fext.lower() in [".part", ".ytdl"]:
                        print("Removing broken file: {}".format(f))
                        os.remove(vidfile)

train_list = None
test_list = None
label2name = None

script_dir = Path(__file__).parent.resolve()
os.chdir(os.path.join(script_dir, "../mmaction2/data/YouTube-Cars"))

with open("train_list.json") as tf:
    train_list = json.loads(tf.read())

with open("test_list.json") as tf:
    test_list = json.loads(tf.read())

with open("label2name.json") as tf:
    label2name = json.loads(tf.read())

shutil.rmtree("train", ignore_errors=True)
shutil.rmtree("test", ignore_errors=True)

os.makedirs("train")
os.makedirs("test")

fetch("train", train_list, label2name)
clean(os.path.join(os.getcwd(), "train"))

fetch("test", test_list, label2name)
clean(os.path.join(os.getcwd(), "test"))
