import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # un peu comme numpy, mais plus adapté à la manipulation de données
import plotly.express as px  # affichage
import sklearn  # pour importer le jeu de données
from sklearn import datasets

digits = sklearn.datasets.load_digits()  # charge jeu de données dans digits

plt.imshow(digits.images[0], cmap=plt.cm.gray_r)
_, axes = plt.subplots(nrows=1, ncols=10, figsize=(20, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r)
    ax.set_title(label)