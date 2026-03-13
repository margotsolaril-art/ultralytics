""" 
Le mode Train est utilisé pour entraîner un modèle YOLO sur un ensemble de données personnalisé. 
Dans ce mode, le modèle est entraîné à l'aide de l'ensemble de données et des hyperparamètres spécifiés. 
Le processus d'entraînement consiste à optimiser les paramètres du modèle afin qu'il puisse prédire avec 
précision les classes et les emplacements des objets dans une image.
"""

from ultralytics import YOLO

""" À partir d'un modèle pré-entrainé """
model = YOLO("yolo26n.pt")  # pass any model type
results = model.train(epochs=5)

""" À partir de zéro """
model = YOLO("yolo26n.yaml")
results = model.train(data="coco8.yaml", epochs=5)

""" Reprendre l'entraînement """
model = YOLO("last.pt")
results = model.train(resume=True)


"""
Trainers use 
L'argument YOLO La classe modèle sert de wrapper de haut niveau pour les classes Trainer. 
Chaque YOLO dispose de son propre trainer, qui hérite de BaseTrainer. Cette architecture permet une 
plus grande flexibilité et personnalisation dans votre flux de travail d'apprentissage automatique.
"""

from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator

# trainer
trainer = DetectionTrainer(overrides={}) # Crée une instance du trainer avec des paramètres vides
trainer.train()  # Lance l'entraînement du modèle
trained_model = trainer.best # Récupère le meilleur modèle entraîné basé sur les performances de validation

# Validator
val = DetectionValidator(args=...) # Crée une instance du validateur avec les arguments nécessaires
val(model=trained_model) # Évalue la performance du modèle entraîné sur un ensemble de validation

# predictor
pred = DetectionPredictor(overrides={})
pred(source=0, model=trained_model) # Fait des prédictions sur la source en utilisant le modèle entraîné

# resume from last weight
overrides = {} # Définit le dictionnaire
overrides["resume"] = trainer.last # Récupère le dernier checkpoint d'entraînement
trainer = DetectionTrainer(overrides=overrides) # Crée un nouveau trainer configuré pour reprendre l'entraînement depuis ce checkpoint 