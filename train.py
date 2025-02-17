import pandas as pd
import numpy as np
from nnet import YapaySinirAgi

def category_encode(etiketler, etiketCesit = 10):
    categorized_etiketler = []
    for i in etiketler:
        vector = np.zeros(etiketCesit)
        vector[i]=1
        #print(vector, "===>=", str(i))
        categorized_etiketler.append(vector)
    return np.array(categorized_etiketler)
        

dataset = pd.read_csv("dataset.csv")

y = dataset["label"]
y = category_encode(y)
x = dataset.drop("label", axis=1)
#print(y)

def egitimVerisi(): pass

egitimVerisi.x = x
egitimVerisi.y = y

yapayzeka = YapaySinirAgi()

yapayzeka.egit(egitimVerisi,150)
yapayzeka.kaydet("kgtu_egitilmis_model_150.h5")