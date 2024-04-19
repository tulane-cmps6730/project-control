import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.getcwd(), "./libraries/repeng"))

from repeng import ControlVector, ControlModel, DatasetEntry

# Define the directory containing the .pt files

directory = "nlp/data/vectors/"

# Iterate through all files in the directory
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".pt"):  # Check if the file is a .pt file
        # Load the .pt object
        cv = torch.load(os.path.join(directory, filename))
        
        cv_data = []
        for i in cv.directions:
            cv_data.append(cv.directions[i])

        # Calculate mean values
        mean_values = np.mean(cv_data, axis=0)
        mean_values_trans = np.mean(cv_data, axis=1)

        # Plot and save the first image
        plt.figure(figsize=(10, 1))
        plt.imshow([mean_values], cmap='seismic', aspect='auto', extent=[0, len(mean_values), 0, 1])
        plt.colorbar(label='Mean Value')
        plt.title('Average Control Vector Across Layers - ' + filename[:-3].replace('_', ' '))
        plt.xlabel('Index X')
        plt.ylabel('Intensity')
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(right=0.7)
        plt.savefig(os.path.join(directory, '../images', filename[:-3] + '_mean_values.png'),bbox_inches="tight")
        plt.close()

        # Plot and save the second image
        plt.figure(figsize=(10, 1))
        plt.imshow([mean_values_trans], cmap='seismic', aspect='auto', extent=[0, len(mean_values_trans), 0, 1])
        plt.colorbar(label='Mean Value')
        plt.title('Average CV Weight Magnitude for Each Layer - ' + filename[:-3].replace('_', ' '))
        plt.xlabel('Index X')
        plt.ylabel('Intensity')
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(right=0.7)  
        plt.savefig(os.path.join(directory, '../images', filename[:-3] + '_mean_values_trans.png'),bbox_inches="tight")
        plt.close()
