#-------------------------------------------------------------------------------
# Name:        generateHeatmap
# Purpose:
#
# Author:      yzlco
#
# Created:     01/12/2019
# Copyright:   (c) yzlco 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import scipy.spatial as sp, scipy.cluster.hierarchy as hc

#http://seaborn.pydata.org/generated/seaborn.clustermap.html
import seaborn as sns; sns.set(color_codes=True)
from matplotlib import rcParams

class generateHeatmap():
    def __new__(self,mtx,deg_names,pag_ids,**kwargs):
        plt.figure(figsize=(5,5))
        # parameters in the heatmap setting 
        width_ratio = 1
        annotationSize = 6
        font_size = 12
        rowCluster = False
        colCluster = False
        
        if 'width_ratio' in kwargs.keys():
            width_ratio = kwargs['width_ratio']      
        if 'annotationSize' in kwargs.keys():
            annotationSize = kwargs['annotationSize']
        if 'rowCluster' in kwargs.keys():
            rowCluster = kwargs['rowCluster']            
        if 'colCluster' in kwargs.keys():
            colCluster = kwargs['colCluster']
            
        outputdir = kwargs['outputdir'] if 'outputdir' in kwargs.keys() else ""
        if deg_names.size > 1:
            #fig, ax = plt.subplots(figsize=(5/len(pag_ids), length))
            # {‘ward’, ‘complete’, ‘average’, ‘single’}
            col_linkage = hc.linkage(sp.distance.pdist(mtx.T), method='average')
            row_linkage = hc.linkage(sp.distance.pdist(mtx), method='average')
        
        # load the color scale using the cm
        #top = cm.get_cmap('Blues_r', 56)
        bottom = cm.get_cmap('Reds', 56)
        newcolors = np.vstack(
                                (
                                    #top(np.linspace(0, 1, 56)),
                                    ([[0,0,0,0.1]]),
                                    bottom(np.linspace(0, 1, 56))
                                )
                             )
        newcmp = ListedColormap(newcolors, name='RedBlue')
        # set the balance point of the expression to 0
        f_max = np.max(mtx)
        f_min = np.min(mtx)
        
        if(abs(f_max)>abs(f_min)):
            Bound=abs(f_max)
        else:
            Bound=abs(f_min)
           
        # figure size in inches
        #rcParams['figure.figsize'] = 3,8.27
        expMtxsDF = pd.DataFrame(mtx)
        expMtxsDF.columns = deg_names
        expMtxsDF.index = pag_ids
        #sns.set(font_scale=1,rc={'figure.figsize':(3,20)})
        
        
        #print(rowCluster == True and int(deg_names.size) > 1)

        #print(int(deg_names.size))
        if(rowCluster == True and colCluster == True and int(deg_names.size) > 1): 
            g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,col_linkage=col_linkage,row_linkage=row_linkage,  yticklabels=True,
                          annot=True,annot_kws={"size": annotationSize})        
        
        elif(rowCluster == True and int(deg_names.size) > 1): 
            g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,col_cluster=False,row_linkage=row_linkage, yticklabels=True,
                          annot=True,annot_kws={"size": annotationSize})
        elif(colCluster == True and int(deg_names.size) > 1): 
            g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,col_linkage=col_linkage,row_cluster=False,  yticklabels=True,
                          annot=True,annot_kws={"size": annotationSize})  
        else:
            if int(deg_names.size) == 1:
                expMtxsDF = expMtxsDF.sort_values(by=list(deg_names), ascending=False)
            g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,row_cluster=False,col_cluster=False,  yticklabels=True,
                          annot=True,annot_kws={"size": annotationSize})  
            #g.ax_row_dendrogram.set_xlim([0,0]) 
        #plt.subplots_adjust(top=0.9) # make room to fit the colorbar into the figure
        ### rotation of labels of x-axis and y-axis
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize= font_size)
        plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize= font_size-2)
        hm = g.ax_heatmap.get_position()
        scale_factor = len(pag_ids)/40
        if scale_factor <  0.5:
            scale_factor = 0.5
        #max_content_length = (40/max([len(pag) for pag in pag_ids]))
        #if max_content_length >10:
        #    max_content_length = 10
        #width_ratio = width_ratio * max_content_length * int(deg_names.size**2)
        #if scale_factor<3 or scale_factor>7:
        #    width_ratio = width_ratio *1.5
        # to change the legends location
        g.ax_heatmap.set_position([hm.x0*scale_factor, hm.y0*scale_factor, hm.width*width_ratio*scale_factor, hm.height*scale_factor])
        col = g.ax_col_dendrogram.get_position()
        g.ax_col_dendrogram.set_position([col.x0*scale_factor, col.y0*scale_factor, col.width*width_ratio*scale_factor, col.height*0.5]) #
        row = g.ax_row_dendrogram.get_position()
        g.ax_row_dendrogram.set_position([row.x0*scale_factor, row.y0*scale_factor, row.width*scale_factor, row.height*scale_factor]) #
        #for i, ax in enumerate(g.fig.axes):   ## getting all axes of the fig object
        #    ax.set_xticklabels(ax.get_xticklabels(), rotation = rotation)
        ### color bar position and title ref: https://stackoverflow.com/questions/67909597/seaborn-clustermap-colorbar-adjustment
        ### color bar position adjustment
        x0, _y0, _w, _h = g.cbar_pos
        g.ax_cbar.set_position([x0, _y0*scale_factor+0.1, row.width*scale_factor, 0.05])
        g.ax_cbar.set_title('-log2 FDR')        
        bottom, top = g.ax_heatmap.get_ylim()
        plt.rcParams["axes.grid"] = False       
        return(plt)
    
if __name__ == '__main__':
    generateHeatmap()
        #plt.rcParams["figure.figsize"] = [20,plt.rcParams["figure.figsize"][1]]
      
        #plt.show()
        
        
        ## create heatmap using imshow
        ##ax = fig.add_subplot(111)
        #im = ax.imshow(mtx, cmap=plt.cm.Blues,aspect="auto")
        ## We want to show all ticks...
        #ax.set_xticks(np.arange(len(deg_names)))
        #ax.set_yticks(np.arange(len(pag_ids)))
        ## ... and label them with the respective list entries
        #ax.set_xticklabels(deg_names)
        #ax.set_yticklabels(pag_ids)
        ##ax.xaxis.tick_top()
        ## create color bar
        #cbar = ax.figure.colorbar(im, ax=ax)
        #cbar.ax.set_ylabel("-log(P)", rotation=-90, va="bottom")
        ## Rotate the tick labels and set their alignment.
        ## plt.setp(ax.get_xticklabels(), rotation=45, ha="right",  rotation_mode="anchor")
        #plt.xticks(rotation=90,fontsize=18)
        #plt.yticks(fontsize=18)
        #for t in ax.xaxis.get_major_ticks():
        #    t.tick1On = False
        #    t.tick2On = False
        #for t in ax.yaxis.get_major_ticks():
        #    t.tick1On = False
        #    t.tick2On = False        
        ##plt.show()
        #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        
        #plt.xlabel('xlabel', fontsize=18)
        #plt.ylabel('ylabel', fontsize=16)
        #plt.savefig("./heatmap.png", dpi=100,transparent = True, bbox_inches = 'tight', pad_inches = 0)        
        # Loop over data dimensions and create text annotations.
##        for i in range(len(pag_ids)):
##            for j in range(len(deg_names)):
##                text = ax.text(j, i, mtx[i, j],ha="center", va="center", color="w")
        #ax.set_title("sample-PAG associations")