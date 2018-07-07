import numpy as np
import matplotlib.pyplot as plt
'''训练集+测试集'''
filetran='train.txt'
filetest='test.txt'

datatran=np.loadtxt(open(filetran,encoding='latin-1'),dtype= str,delimiter='\t')
datatest=np.loadtxt(open(filetest,encoding='latin-1'),dtype= str,delimiter='\t')

tranX=datatran[:,:4]
tranNum=datatran.shape[0]
tranY=datatran[:,-1].reshape(tranNum,1)

testX=datatest[:,:4]
testNum=datatest.shape[0]
testY=datatest[:,-1].reshape(testNum,1)

tranMean=np.mean(tranX,0)
tranSig=np.std(tranX,0,ddof=1)

tranX-=np.tile(tranMean,(tranNum,1))
tranX/=np.tile(tranSig,(tranNum,1))

testX-=np.tile(tranMean,(testNum,1))
testX/=np.tile(tranSig,(testNum,1))

def costFunction(x,y,theta):
    m=y.shape[0]
    cost = np.sum((x.dot(theta)-y)**2)/(2*m)
    return cost

tranX=np.hstack((np.ones((tranNum,1)),tranX))
theta=np.array([1,1,1,1,1])
cost=costFunction(tranX,tranY,theta)
print("tran is cost=",cost)

def desgident(x,y,theta,a=0.01,iter_num=1000):
    m=y.shape[0]
    result=[]
    change=np.zeros(iter_num)
    for i in range(iter_num):
        cost=costFunction(x,y,theta)
        change[i]=cost
        deal=x.T.dot(x.dot(theta)-y)
        theta-=(a/m)*deal
    result.append(theta)
    result.append(change)
    return result

theta=np.zeros((5,1))
result=desgident(tranX,tranY,theta)
print("预测的theta=",result[0].T)

tranTheta=result[0]
testX=np.hstack((np.ones((testNum,1)),testX))
twoValue=testX.dot(tranTheta)
difference=np.hstack((twoValue,testY,twoValue-testY))
print("预测值-实际值-差值")
print(difference)
plt.scatter(twoValue,testY,marker='o',c='r')
plt.show()

