import torch
import tensorflow as tf
from ultralytics import YOLO



# =======================================================================================================================

tf.config.list_physical_devices('GPU')
print()

print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
print(f"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}")
device = torch.device("mps")


# =======================================================================================================================
# Load a model
model = YOLO('/Users/bagsangbeom/PycharmProjects/DMS_project/weights/best (12월 16일).pt')  # build a new model from YAML
detector = model.predict(source=0, show=True, conf=0.4, save=True)