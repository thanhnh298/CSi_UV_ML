import pandas as pd
import os
from os import listdir
from os.path import join


class LoadDataTrain:

    file_path = join(os.getcwd() + '\\'+'training-data', listdir(os.getcwd() + '\\'+'training-data')[0])   

    def __init__(self, file_path):
        self.file_path = file_path
    
    @classmethod
    def getFilePath(cls):
        return cls.file_path

    def getData(self):
        return pd.read_csv(self.file_path)

class LoadDataInput:

    file_path = join(os.getcwd() + '\\'+'input-data', listdir(os.getcwd() + '\\'+'input-data')[0])   

    def __init__(self, file_path):
        self.file_path = file_path
    
    @classmethod
    def getFilePath(cls):
        return cls.file_path

    def getData(self):
        return pd.read_csv(self.file_path)