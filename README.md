# Project Description
[Project Report](https://medium.com/@pawelsendyk/generalizable-real-world-classifiers-for-computer-vision-4230d1cb0b82)
# Setup
Run `pip install -r requirements.txt` to install all the dependencies.
# Running notebooks
From the cloned repo, first run `./data/get_data.sh` and then `./setup` to make the data directory compatible with custom dataloaders.
# Testing
Run  `python test_submission.py [eval.csv]` where `[eval.csv]` is your file with columns: Image_id (int), image_path (str), image_height (int), image_width (int), image_channels (int). This will produce a file eval_classified.csv containing an id and predicted class for each image.
