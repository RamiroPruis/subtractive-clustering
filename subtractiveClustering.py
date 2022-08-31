from select import select
from traceback import print_tb
import numpy as np
import numpy.typing as npt
import scipy as sp


class SubClustering:

    def __init__(self,Ra:float,Rb:float,AR=0.5):
        self.Ra = Ra
        self.Rb = Rb
        self.AR = AR    
        


    def fit(self,data: npt.NDArray):

        data_n = data.shape[0]
        pot_matrix = self.init_potencials(data)
        self.centers = np.empty((0,data.shape[1]))

        max_ant = 0
        max = np.max(pot_matrix)


        while(max > self.AR*max_ant):
            
            center = data[np.argmax(pot_matrix)]
            self.centers = np.vstack((self.centers,center))
            pot_matrix = self.recalc_potentials(pot_matrix,max,center,data)

            max_ant = max
            max = np.max(pot_matrix)
            
        return self
        


    def init_potencials(self,data: npt.NDArray) -> npt.NDArray:
        return np.sum(np.exp(np.negative(sp.spatial.distance_matrix(data,data)/((self.Ra/2)**2))),axis=1)

    
    


    def recalc_potentials(self,pot_matrix,max_pot,center,data):
        return pot_matrix - max_pot*np.exp(np.negative(np.linalg.norm(data-center,axis=1)/((self.Rb/2)**2)))

        
