import pandas as pd
dfwhite = pd.read_csv("winequality-white.csv",sep=';')
dfred = pd.read_csv("winequality-red.csv")
# dfonline = pd.read_csv("http://www.stat.ufl.edu/~winner/data/airq402.txt")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(len(dfwhite),len(dfred)," this is the number of values from initail data")

#first remove duplicate data
dfwhitenew = dfwhite.drop_duplicates()
dfrednew = dfred.drop_duplicates()

print(len(dfwhitenew), len(dfrednew) ," this is the number of values from after deleting duplicate data")

#then drop N/A
dfwhite1 = dfwhitenew.dropna()
dfred1 = dfrednew.dropna()

print(len(dfwhite1),len(dfred1)," this is the number of rows after dropping NA values")

#now we split into train and test
whitetrain = dfwhite1.sample(frac=0.8,random_state=6969)
whitetest = dfwhite1.drop(whitetrain.index)

redtrain = dfred1.sample(frac=0.8,random_state=6969)
redtest = dfred1.drop(redtrain.index)

print(len(whitetrain),len(whitetest))
print(len(redtrain),len(redtest))


X = whitetrain.drop(columns=['quality'])
Y = whitetrain['quality']
print(X)
print(Y)

scaler = StandardScaler()
x_new = scaler.fit_transform(X)
print(x_new)
