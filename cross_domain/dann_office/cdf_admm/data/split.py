import os
import shutil
import numpy as np
from utils.options_office import args

class office31():
    def __init__(self, train_ratio = 0.8, random_seed = 1):
        super(office31, self).__init__()
        self.train_ratio = train_ratio
        self.random_seed = random_seed

    def forward(self, dir_path):
        if not os.path.isdir('data/' + dir_path):
            os.makedirs('data/' + dir_path)
        
            datasets = os.listdir('data/office31')
            
            for dataset in datasets:
                orgPATH = 'data/office31/' + dataset + '/images/'
                newPATH = 'data/office31_split/' + dataset + '/images/'
                os.makedirs(newPATH + 'train/')
                os.makedirs(newPATH + 'test/')
                
                categories = os.listdir(orgPATH)
                
                for category in categories:
                    os.makedirs(newPATH + 'train/' + category + '/')
                    os.makedirs(newPATH + 'test/' + category + '/')
                    
                    files = os.listdir(orgPATH + category + '/')
                    num_files = len(files)
                    num_train_files = int(num_files * self.train_ratio)
                    
                    np.random.seed(self.random_seed)
                    train_idx = np.random.choice(np.arange(num_files), num_train_files, replace = False)
                    train_files = np.array(files)[train_idx].tolist()
                    test_files = list(set(files) - set(train_files))
           
                    for train_file in train_files:
                        orgFilePATH = orgPATH + category + '/' + train_file
                        newFilePATH = newPATH + 'train/' + category + '/' + train_file
                        shutil.copyfile(orgFilePATH, newFilePATH)
                        
                    for test_file in test_files:
                        orgFilePATH = orgPATH + category + '/' + test_file
                        newFilePATH = newPATH + 'test/' + category + '/' + test_file
                        shutil.copyfile(orgFilePATH, newFilePATH)
        else:
             print('data/office31_split/ has already exists.')

   