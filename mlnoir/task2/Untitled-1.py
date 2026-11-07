# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns



# %%
c1 = np.random.normal(loc=[2, 3, 8], scale=[1, 1.5, 1], size=(100, 3))
c2 = np.random.normal(loc=[8, 3, 2], scale=[1, 1, 1.5], size=(100, 3))
c3 = np.random.normal(loc=[5, 8, 5], scale=[1.5, 1, 1], size=(100, 3))
X = np.vstack((c1, c2, c3)) 
y = [0]*100 + [1]*100 + [2]*100
Y = np.eye(3)[y]
print(X.shape,Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# %%
class Logisticreg:
    def __init__(self,lr=0.1,epoch=1000):
        self.lr = lr
        self.epoch = epoch
    def softmax(self,Z):
        ez = np.exp(Z - np.max(Z,axis=1,keepdims=1))
        return ez / np.sum(ez,axis=1,keepdims=1)
    def fit(self,X,Y,loss='ce'):
        self.W = np.zeros((X.shape[1],Y.shape[1]))
        self.b = np.zeros((Y.shape[1]))
        losses = []
        n_samples = X.shape[0]
        for _ in range(self.epoch):
            Z = X @ self.W + self.b
            A = self.softmax(Z)
            A1 = np.clip(A,1e-15,1-1e-15)
            if loss=='ce':
                loss = -np.mean(np.sum(Y * np.log(A1),axis=1))
            elif loss =='kl':
                Y1 = np.clip(Y,1e-15,1)
                loss = np.mean(np.sum(Y * np.log(Y1/A1),axis=1))
            losses.append(loss)
            L = A - Y
            dw = (1/n_samples) * (X.T @ L)
            db = (1/n_samples) * np.sum(L,axis=0)

            self.W -= self.lr * dw
            self.b -= self.lr * db
        return losses
    def predict(self,X):
        Z = X@self.W + self.b
        A = self.softmax(Z)
        return np.argmax(A,axis=1)

# %%
model1 = Logisticreg(lr=0.1,epoch=1000)
model2 = Logisticreg(lr=0.1,epoch=1000)

losses_ce = model1.fit(X_train, Y_train, loss='ce')
losses_kl = model2.fit(X_train, Y_train, loss='kl')

plt.figure(figsize=(10,4))
plt.plot(losses_ce,label="ce",color='red',linewidth=4)
plt.plot(losses_kl,label="kl",color='blue',linewidth=2)
plt.title("loss compare")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


# %%
prediction = model1.predict(X_test)
trueclass = np.argmax(Y_test,axis=1)
print(classification_report(trueclass,prediction,target_names=['Class 0', 'Class 1', 'Class 2']))

cn = confusion_matrix(trueclass,prediction)
print(cn)
sns.heatmap(cn,annot=True)
plt.show()



