import pandas as pd
"""
Epsilon values obtained with elbow method

BATSE: 
    T90-Hrd: 1.0

FERMI
    T90-Hrd: 1.0

SWIFT:
    T90-Hrd: 0.7
"""

epsilons = {'batse': pd.DataFrame({'t90': [0], 'hrd': [0], 't90i': [0], 't90_hrd': [1.0], 't90i_hrd': [0]}), 
            'fermi': pd.DataFrame({'t90': [0], 'hrd': [0], 't90i': [0], 't90_hrd': [1.0], 't90i_hrd': [0]}), 
            'swift': pd.DataFrame({'t90': [0], 'hrd': [0], 't90i': [0], 't90_hrd': [0.7], 't90i_hrd': [0]})}
