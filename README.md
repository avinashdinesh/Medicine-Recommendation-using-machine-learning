import os
HOME = os.getcwd()
print(HOME)

from google.colab import drive
drive.mount('/content/drive')

!pip install ultralytics==8.0.*


from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image

from google.colab import drive
drive.mount('/content/gdrive')
!ln -s /content/gdrive/My\ Drive/ /mydrive
!ls /mydrive

!mkdir -p {HOME}/datasets
%cd {HOME}/datasets

!unzip /content/gdrive/MyDrive/wastev8.zip

%cd /content

!yolo task=detect mode=train model=yolov8s.pt data=/content/datasets/data.yaml epochs=10 imgsz=800 plots=True

import os
from ultralytics import YOLO

# Define paths
HOME = os.getcwd()
trained_model_path = os.path.join(HOME, 'runs', 'detect', 'train', 'weights', 'best.pt')  # Update with the correct path to your trained model
data_yaml_path = os.path.join(HOME, 'datasets', 'data.yaml')  # Path to your data.yaml file

# Load the trained model
model = YOLO(trained_model_path)

# Evaluate the model on the validation dataset
# Add verbose=True to get detailed output during validation
results = model.val(data=data_yaml_path, verbose=True)

# Check if results is None before accessing its elements
if results is not None:
    # Print evaluation metrics
    print(f"Precision: {results['metrics/precision']:.4f}")
    print(f"Recall: {results['metrics/recall']:.4f}")
    print(f"mAP@50: {results['metrics/mAP_50']:.4f}")
    print(f"mAP@50-95: {results['metrics/mAP_50-95']:.4f}")
else:
    print("Validation failed. Check data path, model path, and data.yaml format.")

  !yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data=/content/datasets/data.yaml save=True plots=True
