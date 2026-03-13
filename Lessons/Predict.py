"""
Le mode Prédiction est utilisé pour faire des prédictions à l'aide d'un modèle YOLO entraîné sur de nouvelles images ou vidéos. 
Dans ce mode, le modèle est chargé à partir d'un fichier de point de contrôle, et l'utilisateur peut fournir des images ou des 
vidéos pour effectuer l'inférence. Le modèle prédit les classes et les emplacements des objets dans les images ou vidéos d'entrée.
"""

import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO("model.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam

""" 
From the source 
Passe des chemins de fichiers ou des identifiants directs
Le modèle charge lui-même les images/vidéos
"""
results = model.predict(source="0")
results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

""" 
From PIL 
Charge l'image en objet PIL Image (en mémoire)
Vous contrôlez le chargement
Utile pour traiter l'image avant la prédiction
"""
im1 = Image.open("bus.jpg")
results = model.predict(source=im1, save=True)  # save plotted images

""" 
From ndarray 
Charge l'image en ndarray (tableau NumPy)
Format utilisé par OpenCV (cv2)
Permet des manipulations matricielles sur les pixels
"""
im2 = cv2.imread("bus.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

""" 
From list of PIL/ndarray 
Passe plusieurs images en une seule prédiction
Peut combiner PIL images et ndarrays dans la même liste
Traite tous les objets en batch
"""
results = model.predict(source=[im1, im2])


""" Use of the results """

# 1 - Return as a list
"""
Results would be a list of Results object including all the predictions by default , but be 
careful as it could occupy a lot memory when there're many images, especially the task is segmentation.
"""
results = model.predict(source="folder")


# 2 - Return as a generator
"""
Results would be a generator which is more friendly to memory by setting stream=True
"""
results = model.predict(source=0, stream=True)

for result in results:
    # Detection
    result.boxes.xyxy  # box with xyxy format, (N, 4)
    result.boxes.xywh  # box with xywh format, (N, 4)
    result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    result.boxes.conf  # confidence score, (N, 1)
    result.boxes.cls  # cls, (N, 1)

    # Segmentation
    result.masks.data  # masks, (N, H, W)
    result.masks.xy  # x,y segments (pixels), List[segment] * N
    result.masks.xyn  # x,y segments (normalized), List[segment] * N

    # Classification
    result.probs  # cls prob, (num_class, )

# Each result is composed of torch.Tensor by default,
# in which you can easily use following functionality:
result = result.cuda()
result = result.cpu()
result = result.to("cpu")
result = result.numpy()