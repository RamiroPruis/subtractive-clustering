from re import sub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from subtractiveClustering import SubClustering


def main():
    data1 = pd.read_table('datosEjemplo.txt',sep='\s+',header=None).to_numpy()
    data2 = pd.read_table('datosReduccion.txt',sep='\s+',header=None).to_numpy()
  
    
    

    Ra = 0.6
    Rb = 0.9
    AR = 0.4
    sub_clust1 = SubClustering(Ra,Rb,AR).fit(data1)
    sub_clust2 = SubClustering(Ra,Rb,AR).fit(data2)

    print("DATOS DIAPOSITIVA")
    print(f"cantidad de centros obtenidos: {len(sub_clust1.centers)}")
    print("centros obtenidos")
    print("----------------")
    print("DATOS REDUCCION")
    print(f"cantidad de centros obtenidos: {len(sub_clust2.centers)}")





if __name__ == '__main__':
    main()