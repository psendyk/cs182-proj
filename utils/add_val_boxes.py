with open('./data/tiny-imagenet-200/val/val_annotations.txt', 'r') as val_annotations:
    for line in val_annotations:
        words = line.split()
        img_src, label, X, Y, H, W = words
        with open ('./data/tiny-imagenet-200/val/{}/{}_boxes.txt'.format(label, label), 'a+') as boxes:
            boxes.write("{} \t {} \t {} \t {} \t {}\n".format(img_src, X, Y, H, W))

