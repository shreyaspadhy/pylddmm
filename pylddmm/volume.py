# Several functions for working with volume data from MRICloud outputs
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def extract_from_txt(filename):
    """
        Extracts left and right volumes with names from MRICloud Output text 
        files

        Parameters
        ----------
        filename: str
            Name of .txt file output from MRICloud

        Returns
        -------
        vol_L: Pandas Dataframe
            Columns [vol_name, vol (mm^3)] for left volumes
        vol_R: Pandas Dataframe
            Columns [vol_name, vol (mm^3)] for right volumes
    """
    lines = open(filename, 'r')
    table = []
    for i, line in enumerate(lines):
        if i >= 247 and i < 523:
            table.append(line.strip().split('\t'))

    table = pd.DataFrame(table)

    vol_list = table.iloc[:, 1:3]

    # print(vol_list)
    vol_L = vol_list[vol_list[1].str.contains('_L')]
    vol_R = vol_list[vol_list[1].str.contains('_R')]

    vol_L = vol_L[vol_L[1] != 'Chroid_LVetc_R']

    vol_L = vol_L.sort_values(by=[1])
    vol_R = vol_R.sort_values(by=[1])

    # Swap SFG and SFG_PFC in right
    a, b = vol_R.iloc[104].copy(), vol_R.iloc[105].copy()
    vol_R.iloc[104], vol_R.iloc[105] = b, a

    # Swap SFWM and SFWM_PFC in right
    a, b = vol_R.iloc[108].copy(), vol_R.iloc[109].copy()
    vol_R.iloc[108], vol_R.iloc[109] = b, a

    return vol_L, vol_R
