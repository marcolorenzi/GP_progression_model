import numpy as np
import sys
sys.path.append('../../')
import GP_progression_model
import matplotlib.pyplot as plt
import DataGenerator


# Import data
X,Y,RID,list_biomarker, group = GP_progression_model.convert_csv("./table_APOEposRID.csv",["Hippocampus","Ventricles","Entorhinal","WholeBrain","ADAS11","FAQ","AV45","FDG"])

# Initialize the model
gp  = GP_progression_model.GP_progression_model(X,Y,10, list_biomarker, group = group)

gp.Optimize(5, Plot = True)

[predictions, expectations] = gp.Predict([X,Y])

gp.Plot_predictions(predictions,group=group,names=RID)

