#-------------------------------------------------------------------------------
# Name:		generateNetwork
# Purpose:
#
# Author:	  yzlco
#
# Created:	 08/01/2020
# Copyright:   (c) yzlco 2020
# Licence:	 <your licence>
#-------------------------------------------------------------------------------
import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
class generateNetwork():
    def __new__(self,carac,pos,PPI):

        top = cm.get_cmap('Blues_r', 56)
        bottom = cm.get_cmap('Reds', 56)
        newcolors = np.vstack((
            top(np.linspace(0, 1, 56)),([[1,1,1,0]]),
            bottom(np.linspace(0, 1, 56))))
        newcmp = ListedColormap(newcolors, name='RedBlue')
        
        ### end color scale ###
        fig = plt.figure(figsize=[50,50])
        ax = fig.add_subplot(111)
        plt.axis('off')

        ### network layer ###
        X=nx.Graph()
        X.add_nodes_from(pos.keys())
        ## edge layout ##
        X.add_edges_from(PPI)
        d = dict(X.degree)

        ### maxiumum boundary
        upper=abs(np.max(carac['myvalue'].tolist()))
        if abs(np.min(carac['myvalue'].tolist()))>upper:
            upper=abs(np.min(carac['myvalue'].tolist()))
        if(upper==0):
            upper=1
        #try:
        nx.draw_networkx_edges(X, pos, edgelist=PPI, edge_color='black', arrows=False, alpha =0.2,node_size=0.1)

        nx.draw_networkx_nodes(X, pos, node_color=carac['myvalue'],# with_labels=False,
                           vmin=-upper, vmax=upper, cmap=newcmp,
                           alpha =0.4,node_size=[int(np.log(i+1)*400) for i in d.values()])

        for n, p in pos.items():
            X.nodes[n]['pos'] = p
            # add text # transform=ax.transAxes,
            ax.text(p[0],p[1],n,verticalalignment='center', horizontalalignment='center',
                    color='black', alpha=1, fontsize=np.log(d[n]+800)/np.log(10)*10)
            
        #except:
        #    print("No edges!")
        #nx.draw(X, pos, node_size=[v * 10 for v in d.values()])
        #print(d)
        #cbar = plt.colorbar()
        #cbar.ax.tick_params(labelsize=50)
        #plt.show()
        #plt.savefig("./network.png", dpi=500,transparent = True)#, bbox_inches = 'tight', pad_inches = 0)      
        return(X)
if __name__ == '__main__':
    generateNetwork()
