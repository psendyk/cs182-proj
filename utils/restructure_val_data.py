import os
import pathlib

with open('./data/tiny-imagenet-200/val/val_annotations.txt') as f:
    for line in f:
        words = line.split()
        img_src, label = words[0], words[1]
        folder_path = "./data/tiny-imagenet-200/val/{}/images".format(label)
        if not os.path.exists(folder_path):
            pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
        os.rename("./data/tiny-imagenet-200/val/images/{}".format(img_src), 
                "{}/{}".format(folder_path, img_src))

