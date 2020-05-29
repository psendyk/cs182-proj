import os

val_dir = './data/tiny-imagenet-200/val/'
for label in os.listdir(val_dir):
    if os.path.isdir(val_dir+label):
        try:
            os.remove('./data/tiny-imagenet-200/val/{}/{}_boxes.txt'.format(label, label))
        except FileNotFoundError:
            pass
