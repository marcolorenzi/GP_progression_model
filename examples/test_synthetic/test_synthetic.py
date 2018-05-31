import numpy as np
import sys
sys.path.append('../../')
import GP_progression_model
import DataGenerator
import matplotlib.pyplot as plt


# Creating synthetic multivariate progressions data with monotonic behaviour
L = 1
k = 0.3
interval = [-15,15]

# Number of biomarkers
Nbiom = 6 
# Number of individuals
Nsubs = 100
# Gaussian observational noise
noise = 0.05

flag = 0
while (flag!=1):
    CurveParam = []
    for i in range(Nbiom):
        CurveParam.append([L,0.8*np.random.rand(),noise])
        if CurveParam[i][1] > 0.0:
            flag = 1

# Generating short-temrm time series with given parameters 

dg = DataGenerator.DataGenerator(Nbiom, interval, CurveParam, Nsubs)

# Creating input data lists

X = []
Y = []

for biom in range(len(dg.ZeroXData)):
    X.append([])
    Y.append([])
    for sub in range(len(dg.ZeroXData[biom])):
        X[biom].append([0])
        Y[biom].append([dg.YData[biom][sub][0]])


# Number of random features for kernel approximation
N=int(10)

# Model instantiation

gp  = GP_progression_model.GP_progression_model(dg.ZeroXData,dg.YData,N)

# Optimize

gp.Optimize(Plot = True)

# Prediction of ground-truth time-shift from individuals' short-term observations 

pred_prob, pred_exp = gp.Predict([dg.ZeroXData,dg.YData])

gp.Plot_predictions(pred_prob,[str(i) for i in range(len(X[0]))])

plt.scatter(pred_exp,dg.OutputTimeShift())
plt.xlabel('Predicted time shift')
plt.ylabel('Ground truth')
plt.title('estimated time-shift vs ground truth')
plt.show()


