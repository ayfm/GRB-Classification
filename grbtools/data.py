import os
import matplotlib as mplot
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from .env import DIR_FIGURES
import pickle
from grbtools import models, env
from sklearn.cluster import DBSCAN
from grbtools import parameters


def estimateEpslionDBSCAN(data, feat1, feat2, catalogue_name):
    data = data.sort_values(by = [feat1, feat2])
    df_differences = pd.DataFrame(columns = ['index', 'distance'])
    for i in range(0, len(data) - 1):
        dist = np.linalg.norm(data.iloc[i] - data.iloc[i+1])
        #df_differences = df_differences.append({'index': str(i), 'distance': dist}, ignore_index = True)
        df_differences['index'].append()
        
    df_differences = df_differences.sort_values(by = ['distance'])
    plt.scatter(df_differences['index'], df_differences['distance'], s= 2)
    plt.title("ELBOW | " + catalogue_name.upper() + " " + feat1 + "-" + feat2)
    plt.grid(color='lightgray')
    plt.xticks([])
    plt.yticks(np.arange(min(df_differences['distance']), max(df_differences['distance'])+1, 0.25))
    plt.savefig(env.DIR_FIGURES + "/" + catalogue_name + "/elbow_plots/" + feat1 + "_" + feat2 + ".pdf")
    
    

def performDBSCAN(data, eps, min_samples):
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    outliers = clustering.labels_
    
    
    
def outliers(data, catalogue_name, feat1, feat2):
    eps = parameters.epsilons[catalogue_name][feat1 + "_" + feat2]
    print("Eps: ", eps)
        
    