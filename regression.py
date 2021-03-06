from numpy import *

def loadDataSet(filename)
	numFeat=len(open(filename).readline().split('\t'))-1
	dataMat=[];labelMat=[]
	fr=open(filename)
	for line in fr.readlines():
		lineArr=[]
		curLine=line.strip().split('\t')
		for i in range(lineArr)
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat

def standRegress(xArr,yArr)
	xMat=mat(xArr);yMat=mat(yArr).T
	xTx=xMat.T*xMat
	if linalg.det(xTx)==0.0
		print "This matrix is singular, cannot do inverse"
		return
	ws=xTx.I*(xMat.T*yMat)
	return ws
	
