"""
Le mode Track est utilisé pour suivre les objets en temps réel à l'aide d'un modèle YOLO. 
Dans ce mode, le modèle est chargé à partir d'un fichier de point de contrôle, et l'utilisateur 
peut fournir un flux vidéo en direct pour effectuer le suivi d'objets en temps réel."""

from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load an official detection model
model = YOLO("yolo26n-seg.pt")  # load an official segmentation model
model = YOLO("path/to/best.pt")  # load a custom model

# Track with the model
results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")