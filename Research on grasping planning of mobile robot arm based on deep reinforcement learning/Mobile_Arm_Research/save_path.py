import pandas as pd

def save_csv(list, name, num):
    #pd.DataFrame(columns=name, data=list).to_csv("../saveImg/{}-num.csv".format(num))
    pd.DataFrame(columns=name, data=list).to_csv("../tests/{}-num.csv".format(num))

if __name__ == "__main__":
    #ls = [[1,2,3], [3, 2, 1], [1, 3, 5]]
    #name = ['id', 'num', 'age']
    ls = [1,3,4]
    name = ['id']
    save_csv(ls, name, 1)