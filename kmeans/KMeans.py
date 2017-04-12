from numpy import *

import urllib
import json
import matplotlib
import matplotlib.pyplot as plt

def loadData(filename):
    dataMat = []
    fr = open(filename)
    arrlines = fr.readlines()
    for line in arrlines :
        curline = line.strip().split("\t")
        fitline = list(map(float,curline))
        dataMat.append(fitline)
    return mat(dataMat)

def calcDist(vecA,vecB):
    return sqrt(sum(power((vecA - vecB),2)));

def randCent(dataMat,k):
    n = shape(dataMat)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minj = min(dataMat[:,j])
        rangej = float( max(dataMat[:,j]) - minj )
        centroids[:,j] = minj + random.rand(k,1)*rangej
    return centroids;

def kMeans(dataMat,k,distMeas = calcDist,createCent = randCent):
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid = createCent(dataMat,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            mindistence = inf;clusterIndex = -1;
            for j in range(k):
                dis = distMeas(dataMat[i,:],centroid[j,:])
                if dis < mindistence:
                    mindistence = dis
                    clusterIndex = j

            if clusterAssment[i,0] != clusterIndex : clusterChanged = True
            clusterAssment[i,:] = clusterIndex,mindistence**2
        for cen in range(k):
            ptsInClust = dataMat[nonzero(clusterAssment[:,0]==cen)[0]]
            centroid[cen:] = mean(ptsInClust,axis=0)

    return centroid,clusterAssment

def biKmeans(dataMat,k,distMeas = calcDist):
    m = shape(dataMat)[0]
    clusterAssignment = mat(zeros((m,2)))
    centroid0 = mean(dataMat,axis=0).tolist()[0]
    cenlist = [centroid0]
    for j in range(m):
        clusterAssignment[j,1] = distMeas(mat(centroid0),dataMat[j,:])**2;

    while (len(cenlist) < k):
        lowestSSE = inf
        for i in range(len(cenlist)):
            ptsInCurrCluster = dataMat[nonzero(clusterAssignment[:,0].A == i)[0],:] #选择一列
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas=distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotsplit = sum(clusterAssignment[nonzero(clusterAssignment[:,0].A != i)[0],1])

            if(sseSplit+sseNotsplit < lowestSSE):
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseNotsplit+sseNotsplit

        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0] = len(cenlist) #新加簇
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0] = bestCentToSplit #划分簇

        cenlist[bestCentToSplit] = bestNewCents[0,:] #对于原本的簇进行赋值
        cenlist.append(bestNewCents[1,:]) #新加簇进行添加

        clusterAssignment[nonzero(clusterAssignment[:,0].A == bestCentToSplit)[0],:] = bestClustAss

    return cenlist,clusterAssignment


def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy


def cluster(numClust=5):
    datList = []
    fileArrlines = open('D:\predPaths_test.txt').readlines()
    for line in fileArrlines:
        lineArr = line.strip().split(',')
        datList.append([float(lineArr[3]), float(lineArr[2])])
    datMat = mat(datList)
    print("读完数据开始计算！")
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()
