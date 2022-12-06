from pingouin import multivariate_normality
import pandas as pd
import numpy as np

data = pd.read_csv("data/train.csv")
x_train = data.iloc[:100,2:4]
print(x_train)

result = multivariate_normality(x_train,alpha=.05)
print(result)

df = pd.DataFrame({'x1':np.random.normal(size=50),
                   'x2': np.random.normal(size=50),
                   'x3': np.random.normal(size=50)})

#print(df)

result2 = multivariate_normality(df,alpha=0.05)
print(result2)