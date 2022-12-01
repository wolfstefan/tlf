#!/usr/bin/python3

import os
import shutil

from pathlib import Path

def create_file_class_index(path, name, class_index, destination):
    print("Creating file class index '{}' at {}...".format(name, destination))
    index_file = open("{}/{}".format(destination, name), "w")

    for directory in sorted([cdir for cdir in os.listdir(path) if os.path.isdir(os.path.join(path, cdir))]):
        cind = class_index.index(directory)
        for dfile in os.listdir(os.path.join(path, directory)):
            fullpath = os.path.join(os.path.join(path, directory), dfile)
            index_file.write("{} {}\n".format(fullpath, cind))
    index_file.close()

def create_file_class_indices(top_directory):
    print("Generating file class indices...")

    train_path = os.path.join(top_directory, "train")
    val_path = os.path.join(top_directory, "validation")
    test_path = os.path.join(top_directory, "test")
    
    train_dir_exists = os.path.exists(train_path) and os.path.isdir(train_path)
    val_dir_exists = os.path.exists(val_path) and os.path.isdir(val_path)
    test_dir_exists = os.path.exists(test_path) and os.path.isdir(test_path)

    class_index = []

    if not train_dir_exists:
        print("Error! Required /train directory was not found. Aborting...")
        return False
    else:
        print("Found /train directory...")
        class_index = sorted([cdir for cdir in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, cdir))])

    if not val_dir_exists:
        print("Optional /validation directory was not found and will be skipped...")
    else:
        print("Found /validation directory...")
        for val_class in sorted(os.listdir(val_path)):
            if val_class not in class_index:
                print("\t- Removing class {} in /validation since it does not exist in /train...".format(val_class))
                shutil.rmtree(os.path.join(val_path, val_class))

    if not test_dir_exists:
        print("Optional /test directory was not found and will be skipped...")
    else:
        print("Found /test directory...")
        for test_class in sorted(os.listdir(test_path)):
            if test_class not in class_index:
                print("\t- Removing class {} in /test since it does not exist in /train...".format(test_class))
                shutil.rmtree(os.path.join(test_path, test_class))

    if train_dir_exists:
        create_file_class_index(train_path, "train_index.txt", class_index, top_directory)
    if val_dir_exists:
        create_file_class_index(val_path, "validation_index.txt", class_index, top_directory)
    if test_dir_exists:
        create_file_class_index(test_path, "test_index.txt", class_index, top_directory)

script_path = Path(__file__).parent.resolve()
cars_path = os.path.join(script_path, "../mmaction2/data/YouTube-Cars")
birds_path = os.path.join(script_path, "../mmaction2/data/YouTube-Birds")

create_file_class_indices(cars_path)
create_file_class_indices(birds_path)