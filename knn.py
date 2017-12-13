import common
from sklearn.neighbors import KNeighborsClassifier
import pickle
import logging
from sklearn.model_selection import cross_val_score

scores = common.attempt(KNeighborsClassifier(), range(1,3))
