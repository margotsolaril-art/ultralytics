"""
Le mode Val est utilisé pour valider un modèle YOLO après son entraînement. 
Dans ce mode, le modèle est évalué sur un ensemble de validation pour mesurer 
sa précision et ses performances de généralisation. Ce mode peut être utilisé 
pour ajuster les hyperparamètres du modèle afin d'améliorer ses performances.
"""

from ultralytics import YOLO

# Load a YOLO model
model = YOLO("yolo26n.yaml")

# Train the model
model.train(data="coco8.yaml", epochs=5)

# Validate 
""" On training data """
model.val()

""" On separate data """
model.val(data="path/to/separate/data.yaml")