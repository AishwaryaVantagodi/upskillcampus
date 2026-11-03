Put your labelled crop/weed dataset here in YOLO format.

Expected:
images/train/    training images
images/val/      validation images
labels/train/    matching YOLO .txt files
labels/val/      matching YOLO .txt files
data.yaml        dataset configuration

Example data.yaml (edit path to your actual absolute path or a correct relative path):
path: C:/full/path/to/Integrated_Agri_Project/dataset/weed_detection
train: images/train
val: images/val
names:
  0: crop
  1: weed

Each image should have a matching label text file with rows:
class_id x_center y_center width height
All box coordinates must be normalized between 0 and 1.

Do not train on unlabelled images or make up labels. If you only have the project description but not the actual 1300 images and labels, obtain or create the labelled dataset first.
