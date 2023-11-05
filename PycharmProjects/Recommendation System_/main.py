import random
import os
import gensim
import numpy as np
import pandas as pd

from DataPreprocessing import preprocess_data
from SVM_Model import SVM_Model
from mergeFiles import mergeFiles
from IncrementalLearning import IncrementalLearning

current_file = __file__
f = os.path.dirname(os.path.abspath(current_file))

if __name__ == '__main__':
    # # Merge all collected files into a single File to process:
    # mergeFiles()

    # # Data Preprocessing to give into SVM model:
    # preprocess_data()

    # SVM Model: Recommend product following customer description
    SVM = SVM_Model()
    # SVM.evaluateModel()
    # SVM.trainModel()
    SVM.loadModel()
    print(SVM.recommend("Ngoại hình bắt mắt, thu hút mọi ánh nhìn"))

    # Incremental Learning for Recommendation System
    # IL = IncrementalLearning()
    # filePath = f + "/additionalData/collectedData.csv"
    # IL.__int__(filePath)
    # result = IL.remove_noiseData()
    # IL.writeNewFile(result)



