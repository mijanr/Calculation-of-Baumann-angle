import glob
import json
import  torch
import  torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import PIL
import random

class Get_Data:
    def __init__(self):
        pass
    def patient_level_split(self,id_path:str=None,split_ratio:float=0.9)->tuple:
        id_path = '../XR_ELBOW'
        ids = os.listdir(id_path)
        random.seed(0)
        random_index = random.sample(range(len(ids)), int(len(ids)*split_ratio))
        train_ids = [ids[i] for i in random_index]
        test_ids = [ids[i] for i in range(len(ids)) if i not in random_index]

        #key_points_path_train
        key_points_path_train = list()
        for i in range(len(train_ids)):
            jsons = glob.glob('../XR_ELBOW/' + train_ids[i] + '/*/*.json')
            key_points_path_train.append([item.split('XR_ELBOW')[1] for item in jsons])
        #flatten the list
        key_points_path_train = [item for sublist in key_points_path_train for item in sublist]

        #key_points_path_test
        key_points_path_test = list()
        for i in range(len(test_ids)):
            jsons = glob.glob('../XR_ELBOW/' + test_ids[i] + '/*/*.json')
            #only take everything after XR_ELBOW and append to the list
            key_points_path_test.append([item.split('XR_ELBOW')[1] for item in jsons])
            
        #flatten the list
        key_points_path_test = [item for sublist in key_points_path_test for item in sublist]


        #add '../XR_ELBOW' to the path
        key_points_path_train = ['../XR_ELBOW' + item for item in key_points_path_train]
        key_points_path_test = ['../XR_ELBOW' + item for item in key_points_path_test]
        
        self.json_path_train = key_points_path_train
        self.json_path_test = key_points_path_test

    
        return key_points_path_train, key_points_path_test
    
    def json_to_data(self, json_paths:list, imageH, imageW):
        images = list()
        labels = list()
        for path in json_paths:
            with open(path) as f:
                data = json.load(f)['shapes']
            labels_dict = {}
            for j in range(len(data)):
                labels_dict[data[j]['label']] = data[j]['points']
            shaft_cntrline = labels_dict['Shaft Centerline']
            tangent = labels_dict['Tangent']
            label = torch.Tensor(np.array(shaft_cntrline+tangent)).float()
            
            img_path = '../MURA-v1.1/train'+path[2:-5]+'.png'
            image = plt.imread(img_path)
            if len(image.shape)>2:
                image = np.mean(image, axis=2)
            oldX, oldY = image.shape[0], image.shape[1]

            image = PIL.Image.fromarray(image)
            image = image.resize((imageH, imageW))
            image = np.array(image)    # (hxw)        
            image = np.expand_dims(image, axis=0)

            #get the ratios
            Rx, Ry = imageH/oldX, imageW/oldY

            #fix the keypoints
            label[:,0] = label[:,0]*Ry
            label[:,1] = label[:,1]*Rx


            labels.append(label)
            images.append(torch.Tensor(image).float())
        return images, labels


    

if __name__ == "__main__":
    get_data = Get_Data()
    key_points_path_train, key_points_path_test = get_data.patient_level_split()
    images, labels = get_data.json_to_data(key_points_path_train, 512, 512)
    print(len(images))