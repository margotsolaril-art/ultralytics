"""
Le mode d'exportation est utilisé pour exporter un modèle YOLO dans un format qui peut être utilisé 
pour le déploiement. Dans ce mode, le modèle est converti dans un format qui peut être utilisé par 
d'autres applications logicielles ou périphériques matériels. Ce mode est utile lors du déploiement du 
modèle dans des environnements de production.
"""

from ultralytics import YOLO

model = YOLO("yolo26n.pt")

""" Export to ONNX format, which can be used for interoperability with other frameworks. """
model.export(format="onnx", dynamic=True)

""" Export to TensorRT engine, which can be used for deployment on NVIDIA devices. """
model.export(format="engine", device=0)



"""
Benchmark
Le mode d'évaluation comparative est utilisé pour profiler la vitesse et la précision de divers formats 
d'exportation pour YOLO. Les benchmarks fournissent des informations sur la taille du format exporté, 
son mAP50-95 métriques (pour la détection d'objets et la segmentation) ou accuracy_top5 métriques (pour 
la classification), et le temps d'inférence en millisecondes par image à travers divers formats 
d'exportation comme ONNX, OpenVINO, TensorRT et autres. Ces informations peuvent aider les utilisateurs 
à choisir le format d'exportation optimal pour leur cas d'utilisation spécifique en fonction de leurs 
exigences en matière de vitesse et de précision.
"""

from ultralytics.utils.benchmarks import benchmark

# Benchmark
benchmark(model="yolo26n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)