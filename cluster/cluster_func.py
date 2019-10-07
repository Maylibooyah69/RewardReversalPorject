

import numpy as np
import pandas as pd
from sklearn import mixture
from sklearn.cluster import KMeans

from sklearn.metrics import calinski_harabasz_score
#from sklearn.metrics import silhouette_score




#________________________GaussianMixture___________________

# fit Gaussian Mixture model given n_components and feature data
def fit_gmm(n_comp,X):
    gmm = mixture.GaussianMixture(n_components=n_comp)
    labels = gmm.fit_predict(X)

    bic = gmm.bic(X)
    aic = gmm.aic(X)

    return gmm,bic,aic,labels


def fit_predict_gmm(n_comp,X):
    gmm = mixture.GaussianMixture(n_components=n_comp)
    label = gmm.fit_predict(X)

    bic = gmm.bic(X)
    aic = gmm.aic(X)
    
    return gmm,bic,aic,label



# fit Gaussian Mixture model from n_comp = 2 to max_comp
# max_comp default = 9
def gmm(X, max_comp=9):
    gmm_list = []
    bic_list = []
    aic_list = []
    label_list = []
    n_comp_list = list(range(2,max_comp))

    for i in range(2,max_comp):
        gmm,bic,aic,labels = fit_gmm(i,X)

        gmm_list.append(gmm)
        bic_list.append(bic)
        aic_list.append(aic)
        label_list.append(labels)

    return gmm_list,bic_list,aic_list,label_list, n_comp_list



# _____________________K_Means_______________________

def fit_kmeans(X,k):
    km =KMeans(n_clusters=k)
    labels = km.fit_predict(X)
    ssd = km.inertia_

    # get silhouette and c-h score
    silhouette = silhouette_score(X,labels)
    ch = calinski_harabasz_score(X,labels)

    return km,ssd,silhouette,ch,labels


def fit_predict_kmeans(X,k):
    km = KMeans(n_clusters=k)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    return centers,labels,km


def kmeans(X,max_k=9):
    km_list = []
    ssd_list = []
    silhouette_list = []
    ch_list = []
    label_list = []
    k_list = list(range(2,max_k))

    for i in range(2,max_k):
        km,ssd,silhouette,ch,labels = fit_kmeans(X,i)

        km_list.append(km)
        ssd_list.append(ssd)
        silhouette_list.append(slhouette)
        ch_list.append(ch)
        label_list.append(labels)

    return km_list,ssd_list,silhouette_list,ch_list,label_list,k_list


#__________________SOM____________________

def som():
    pass




















