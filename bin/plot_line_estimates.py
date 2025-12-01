#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skimage as ski

def stems( line_estimates ):
    error_est = lest['Est']-lest['GTCnt']
    error_pred = lest['PredCnt']-lest['GTCnt']

    fig, ax = plt.subplots()
    est = plt.stem( x, error_est, markerfmt='or', label="Fourier Estimate (error wr/ GT)")
    pred = plt.stem( x, error_pred, label="NN Prediction (error wr/ GT)", markerfmt='og')
    gt = plt.plot([0,np.max(x)],[0,0], color='red')
    #lest['Img'][ np.abs(lest['PredCnt']-lest['GTCnt']) > 50 ]
    ax.set_ylabel("Line count")
    ax.set_xlabel("Charter images.")
    ax.legend()
    plt.show()

def imgs( line_estimates ):
    
   
    error_est = lest['Est']-lest['GTCnt']
    error_pred = lest['PredCnt']-lest['GTCnt']
    keep=np.abs(error_pred) > 50
    img_to_box = list(zip(lest['Img'][keep],np.array(lest[['l','t','r','b']][keep])))
    #for img, box_array in img_to_box:
    #    l,t,r,b = box_array
    #    img_hwc = ski.io.imread( img )
    #    crop_hwc = img_hwc[t:b,l:r]
    #    plt.imshow( crop_hwc )
    #    plt.show()

    fig, ax = plt.subplots(2,3)
    imgs = ('900da0e1260c48cd89af8448730fa86c_crop.png', 
            '995909b220d7c74ab594ea3eebda6de4_crop.png',
            'a9c1ee789b7ac45428ce23cc44a03dfb_crop.png')
    indexes = (11,16,38)
    bar_labels = ('GT', 'Fourier-Estimate', 'NN Pred')
    bar_colors = ('tab:red', 'tab:blue', 'tab:orange')

    for col in range(3):
        img_hwc=ski.io.imread( imgs[col] )
        ax[0,col].imshow( img_hwc )
        ax[0,col].set_xlabel(imgs[col].replace('_crop','_r0'))
        #ax[1,col].stem( range(3), lest[['GTCnt','Est','PredCnt']].loc[indexes[col]] )
        ax[1,col].bar( bar_labels, lest[['GTCnt','Est','PredCnt']].loc[indexes[col]], label=bar_labels, color=bar_colors)
    ax[1,0].set_ylabel('Line count')
    #gt = plt.plot([0,np.max(x)],[0,0], color='red')
    #est = plt.stem( x, error_est, markerfmt='or', label="Fourier Estimate (error wr/ GT)")
    #pred = plt.stem( x, error_pred, label="NN Prediction (error wr/ GT)", markerfmt='og')
    #lest['Img'][ np.abs(lest['PredCnt']-lest['GTCnt']) > 50 ]
    #ax.set_ylabel("Line count")
    #ax.set_xlabel("Charter images.")
    #ax.legend()
    plt.show()



lest = pd.read_csv('line_estimates.tsv', sep="\t", header=0)
#stems(lest)

imgs(lest)


