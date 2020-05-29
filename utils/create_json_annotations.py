import os
import json

train_path = './data/tiny-imagenet-200/train/'
val_path = './data/tiny-imagenet-200/val/'
data = []

for path in [train_path, val_path]:
    for label in os.listdir(path):
        if label == "images":
            continue
        if os.path.isdir(path+label):
            with open(path+label+'/{}_boxes.txt'.format(label), 'r') as boxes:
                for line in boxes:
                    img_src, X, Y, H, W = line.split()
                    X, Y, H, W = list(map(int, [X, Y, H, W]))
                    data.append([{'img_src': img_src, 'label': label, 'boxes': [X, Y, H, W]}])
    with open(path+'annotations.json', 'w') as f:
        json.dump(data, f)
    data = []
