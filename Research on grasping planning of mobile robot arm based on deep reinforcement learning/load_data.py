import pandas as pd
import numpy as np

class Sample_data:
    def __init__(self) -> None:
        self.data = pd.read_csv("../tests/test_data_3.csv")
    
    def Sample(self,index):
        dis = np.array(self.data['dis'])
        tha = np.array(self.data['tha'])
        cup = []
        cup.append(dis[index])
        cup.append(tha[index])
        return cup
    

if __name__ == "__main__":
    #ls = [[1,2,3], [3, 2, 1], [1, 3, 5]]
    #name = ['id', 'num', 'age']
    # data = pd.read_csv("./tests/test_data_1.csv")
    # test = Sample_data(data)
    # print(test.Sample(10))
    pass