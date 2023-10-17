import streamlit as st
from streamlit_agraph import agraph, TripleStore, Config
import pandas as pd
import requests
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import generateheatmap as Heatmap
import generateNetwork as Network
import numpy as np
import plotly.tools
import base64
from streamlit_agraph import agraph, Node, Edge, Config
import plotly
import math
import scanpy as sc
import glob
import anndata

### multiple tabs load style ###
### ref: https://github.com/streamlit/streamlit/issues/233 ###

#st.markdown(
#    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
#    unsafe_allow_html=True,
#)
#query_params = st.experimental_get_query_params()
tabs = ["Data","Step1","Step2","Step3","Step4","Step5"]
#if "tab" in query_params:
#    active_tab = query_params["tab"][0]
#else:
#    active_tab = "Data"
#
#if active_tab not in tabs:
#    st.experimental_set_query_params(tab="Data")
#    active_tab = "Data"


# color mapping of the gene expression #
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import scipy.spatial as sp, scipy.cluster.hierarchy as hc

#http://seaborn.pydata.org/generated/seaborn.clustermap.html
import seaborn as sns; sns.set(color_codes=True)
from matplotlib import rcParams

st.set_option('deprecation.showPyplotGlobalUse', False)

def generateheatmap(mtx,deg_names,pag_ids,**kwargs):
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


# color in hex_map format
colorUnit = 56
top = cm.get_cmap('Blues_r', colorUnit)
bottom = cm.get_cmap('Reds', colorUnit)
newcolors = np.vstack((
    top(np.linspace(0, 1, 56)),([[1,1,1,0]]),
    bottom(np.linspace(0, 1, 56))
))
newcmp = ListedColormap(newcolors, name='RedBlue')
hex_map = [matplotlib.colors.to_hex(i, keep_alpha=True) for i in newcolors]


#st.title('GBM-PDX Data Analysis in U01 Project')
st.title('Interactive Single Cell Analysis')
st.header('An online interactive analytical platform for single cell functional genomic downstream analysis')
st.markdown('*Zongliang Yue, Robert S. Welner, and Jake Chen*')


st.sidebar.subheader('Code source')
link = 'The scanpy python script is [https://scanpy-tutorials.readthedocs.io/en/latest/index.html](https://scanpy-tutorials.readthedocs.io/en/latest/index.html)'
st.sidebar.markdown(link, unsafe_allow_html=True)



#st.sidebar.text("1.NHBE: Primary human lung epithelium.\n2.A549: Lung alveolar.\n3.Calu3:The transformed lung-derived Calu-3 cells.\n4.Lung: The lung samples.\n5.NP: The nasopharyngeal samples.\n6.PBMC: The peripheral blood mononuclear cell.\n7.Leukocyte: The leukocytes.\n8.hiPSC:Human induced pluripotent stem cell-derived cardiomyocytes\n9.Liver Organoid.\n10.Pancreas Organoid")
workingdir = st.sidebar.selectbox(
    'select a pre-processed single cell dataset::',
    tuple(['mouse_NK']),key='workingdir'
    )
link = "1.mouse_NK is natural killer cells from mouse samples. (confidential)"
st.sidebar.markdown(link, unsafe_allow_html=True)

st.sidebar.markdown('You selected `%s`' % workingdir)





#https://github.com/streamlit/streamlit/issues/400
# get download link

#@st.cache(allow_output_mutation=True)
def get_table_download_link(df, **kwargs):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=True, sep ='\t')
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    prefix = "Download txt file for "
    if("fileName" in kwargs.keys()):
        prefix += kwargs['fileName']
    href = f'<a href="data:file/csv;base64,{b64}" download="'+kwargs['fileName']+'\.txt">'+prefix+'</a>'
    return(href)




def load_files(file_list):
    #file_list = glob.glob(os.path.join(output_dir, "*"))
    i=0
    for file in file_list:
        adata = sc.read_h5ad(file)
        if i == 0:
            st.session_state['adata_merge'] = adata
            i += 1
        else:
            st.session_state['adata_merge'] = anndata.concat([st.session_state['adata_merge'],adata],index_unique=None)
            i += 1
        del(adata)
        print(i)

def chunk_array(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):
        yield arr[i:i + chunk_size]


# Return GBM treatment data as a data frame.
@st.cache(allow_output_mutation=True)
def load_h5ad_file(workingdir):
    #df = pd.read_csv('SampleTreatment.txt',sep="\t")
    #adata_merge = sc.read_h5ad('input/'+workingdir+'/'+'scanpy_adata_merge_15249_unregress.h5ad')
    #adata_merge = load_files('input/'+workingdir+'/scanpy_adata_merge_15249_unregress/')
    ### load the chunked ###
    output_dir = 'input/'+workingdir+'/scanpy_adata_merge_15249_unregress/'
    file_list = glob.glob(os.path.join(output_dir, "*"))
    load_files(file_list[0:4])

    ######
    description = pd.read_csv('input/'+workingdir+'/'+'description.txt',sep="\t")
    cellpop = pd.read_csv('input/'+workingdir+'/'+'cellpop.txt',sep="\t")
    return(description,cellpop)

# Call PAGER REST API to perform hypergeometric test and return enriched PAGs associated with given list of genes as a data frame.
# See pathFun() in PAGER R SDK at https://uab.app.box.com/file/529139337869.
@st.cache(allow_output_mutation=True)
def run_pager(genes, sources, olap, sim, fdr):
	# Set up the call parameters as a dict.
	params = {}
	# Work around PAGER API form encode issue.
	if(len(genes)!=0):
		#print(genes)
		params['genes'] = '%20'.join(genes)
	else:
		params['genes'] = ''
	params['source'] = '%20'.join(sources)
	params['type'] = 'All'
	params['sim'] = sim
	params['olap'] = olap
	params['organism'] = 'All'
	params['cohesion'] = '0'
	params['pvalue'] = 0.05
	params['FDR'] = np.float64(fdr)
	params['ge'] = 1
	params['le'] = 2000

	response = requests.post('http://discovery.informatics.uab.edu/PAGER/index.php/geneset/pagerapi', data=params)
#	print(response.request.body)
	return pd.DataFrame(response.json())

# gene network in PAG
#@st.cache(allow_output_mutation=True)
def run_pager_int(PAGid):
	response = requests.get('http://discovery.informatics.uab.edu/PAGER/index.php/pag_mol_mol_map/interactions/'+PAGid)
	return pd.DataFrame(response.json())

# pag_ranked_gene in PAG
#@st.cache(allow_output_mutation=True)
def pag_ranked_gene(PAGid):
	response = requests.get('http://discovery.informatics.uab.edu/PAGER/index.php/genesinPAG/viewgenes/'+PAGid)
	return pd.DataFrame(response.json()['gene'])

# generate force layout
#@st.cache(allow_output_mutation=True)
def run_force_layout(G):
    pos=nx.spring_layout(G, dim=2, k=None, pos=None, fixed=None, iterations=50, weight='weight', scale=1.0)
    return(pos)




#st.header('Query Clinical Data')
#st.markdown("These data are read from U-BRITE's *treament* programmatically from a secure call the *Unified Web Services* (UWS) API at http://ubritedvapp1.hs.uab.edu:8080/UbriteServices/getalli2b2demographics?requestorid=rdalej&cohortid=27676&format=csv.")
#clinical_data_load_state = st.text('Loading data ... ')
#clinical_data = load_clinical_data()
#clinical_data_load_state.text('Loading data ... done!')
#st.write(clinical_data)

###############
description,cellpop = load_h5ad_file(workingdir)


### multiple tab show tabs ###
#li_items = "".join(
#    f"""
#    <li class="nav-item">
#        <a class="nav-link{' active' if t==active_tab else ''}" href="/?tab={t}" target="_self">{t}</a>
#    </li>
#    """
#    for t in tabs
#)
#tabs_html = f"""
#    <ul class="nav nav-tabs">
#    {li_items}
#    </ul>
#"""
#st.markdown(tabs_html, unsafe_allow_html=True)
#st.markdown("<br>", unsafe_allow_html=True)
#############
data, tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)
adata_merge = st.session_state['adata_merge']
with data:
#if active_tab == "Data":
    st.header('Dataset description')
    st.table(description)
    st.markdown(get_table_download_link(description, fileName = " "+workingdir+' sample description'), unsafe_allow_html=True)  
    st.write("These are the samples, and we performed merged version in union them, please confirm to continue the analysis")
    
    ###############
    st.header('Preprocessed quality control')
    st.markdown("These results based on the min_genes=200, and min_cells=3.")
    #st.markdown("See source code for pipeline in *Source Code Repository* at https://gitlab.rc.uab.edu/gbm-pdx/deseq2-rnaseq.")
    #st.pyplot(sc.pl.violin(adata_merge, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
    #             jitter=0.4, multi_panel=True))
    st.table(cellpop)
    st.markdown(get_table_download_link(cellpop, fileName = " "+workingdir+' cell population'), unsafe_allow_html=True) 
    st.write("After merging the three tissue samples, there are 44201 cells and 15249 samples.")
    #degs = load_deg_results()


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

pct =  adata_merge.obs[['leiden','sample','n_genes']].groupby(['leiden','sample']).count()
pct = pct.reset_index()  
colors_custm = ['steelblue','darkorange','green']
leiden_max = max(pct['leiden'].astype('int')) + 1

with tab1:
    ###############
    st.header('Section 1: Show the single cell map using t-SNE* or UMAP* dimensional reduction')
    method = st.selectbox(
    'method',
    ("tSNE","UMAP"),
    key="method_box"
    )
    sel_cluster = st.multiselect('Select clusters',
        tuple([str(leiden_idx) for leiden_idx in list(range(0,leiden_max))]),
        tuple([str(leiden_idx) for leiden_idx in list(range(0,leiden_max))])
    )
    adata_merge_filtered = adata_merge[adata_merge.obs[adata_merge.obs.leiden.isin(sel_cluster)].index]
    sc.tl.dendrogram(adata_merge_filtered,groupby="leiden")
    if method == "tSNE":
        #fig, axs = plt.subplots(1, 2, figsize=(8,4),constrained_layout=True)
        st.pyplot(sc.pl.tsne(adata_merge_filtered, color="sample", title=" tSNE",frameon=True, #legend_loc='on data'
                            ))
        st.pyplot(sc.pl.tsne(adata_merge_filtered, color="leiden", title=" tSNE", add_outline=True, #legend_loc='on data',
                   legend_fontsize=12, legend_fontoutline=2,frameon=True))

    elif method == "UMAP":
        #fig, axs = plt.subplots(1, 2, figsize=(8,4),constrained_layout=True)
        st.pyplot(sc.pl.umap(adata_merge_filtered, color="sample", title=" UMAP",frameon=True, #legend_loc='on data'
                            ))
        st.pyplot(sc.pl.umap(adata_merge_filtered, color="leiden", title=" UMAP", add_outline=True, #legend_loc='on data',
                   legend_fontsize=12, legend_fontoutline=2,frameon=True))

    #import plotly.express as px
    #fig = px.scatter(x=adata.obsm['X_tsne'][:,0], y=adata.obsm['X_tsne'][:,1],color=adata.obs['bulk_labels'])
    #fig.show()

    # https://plotly.com/python/pie-charts/#basic-pie-chart-with-gopie
    vec = []
    for i in range(0,int(np.ceil(leiden_max/3))):
        vec.append(np.repeat({"type": "domain"},3).tolist())
    fig = make_subplots(rows=int(np.ceil(leiden_max/3)), cols=3, specs=vec,
                       subplot_titles=["c"+str(j) for j in list(range(0, leiden_max))]
                       )
    for i in sel_cluster:
        subpct= pct.where(pct['leiden'] == str(i))
        fig.add_trace(
                        go.Pie(values=subpct['n_genes'], labels=subpct['sample'],sort=False,#marker_colors=subpct['sample'],                  
                              ),
                      row = int(np.ceil((int(i)+1)/3)),
                      col = (int(i)+1-1)%3+1
        )
    fig.update_traces(hoverinfo='label+percent+value', textinfo='percent', textfont_size=10,
                      marker=dict(colors=['steelblue','darkorange','green'], line=dict(color='#000000', width=2)))
    st.plotly_chart(fig)

with tab2:
    ###############
    st.header('Section 2: Show the marker expression mapping')
    
    marker = pd.read_csv('input/'+workingdir+'/'+'marker.txt',sep="\t")
    st.table(marker)
    st.markdown(get_table_download_link(marker, fileName = " "+workingdir+' marker'), unsafe_allow_html=True) 
    markers = ["Itgam","Cd27","Klrb1c","Il2rb"]
    if method == "tSNE":
        st.pyplot(sc.pl.tsne(adata_merge_filtered, color=markers, s=50, frameon=False, vmax='p99',ncols = 2,cmap="viridis"))
    elif method == "UMAP":    
        st.pyplot(sc.pl.umap(adata_merge_filtered, color=markers, s=50, frameon=False, vmax='p99',ncols = 2,cmap="viridis"))
    
    marker = st.selectbox(
    'marker',
    tuple(adata_merge_filtered.var_names.sort_values()),
        key="marker_box"
        )
    if method == "tSNE":
        st.pyplot(sc.pl.tsne(adata_merge_filtered, color=[marker], s=50, frameon=False, vmax='p99',cmap="viridis"))
        st.pyplot(sc.pl.violin(adata_merge_filtered, [marker], groupby='leiden'))
        #sc.pl.violin(adata_merge_filtered, [marker], groupby='leiden')
    elif method == "UMAP":    
        st.pyplot(sc.pl.umap(adata_merge_filtered, color=[marker], s=50, frameon=False, vmax='p99',cmap="viridis"))
        st.pyplot(sc.pl.violin(adata_merge_filtered, [marker], groupby='leiden'))
        #sc.pl.violin(adata_merge_filtered, [marker], groupby='leiden')
    
###############
# return the cluster comparison using the differeially expressed gene analysis 
@st.cache(allow_output_mutation=True)
def compute_DEG(cluster_name,selected_cluster,referece_cluster):
    sc.tl.rank_genes_groups(adata_merge_filtered, cluster_name, groups=[str(selected_cluster)],reference=str(referece_cluster),
                            method='wilcoxon',key_added = "wilcoxon")

@st.cache(allow_output_mutation=True)    
def get_wilcoxon_result(adata_merge_filtered,selected_cluster):
    res_pd = pd.DataFrame()
    method_name = 'wilcoxon'
    #st.table(pd.DataFrame(adata_merge_filtered.uns['rank_genes_groups']['names']))
    res_pd['names'] = pd.DataFrame(adata_merge_filtered.uns[method_name]['names'])[str(selected_cluster)].values#.str.decode('utf-8') 
    res_pd['scores'] = pd.DataFrame(adata_merge_filtered.uns[method_name]['scores'])[str(selected_cluster)].values
    res_pd['logfoldchanges'] = pd.DataFrame(adata_merge_filtered.uns[method_name]['logfoldchanges'])[str(selected_cluster)].values
    res_pd['pvals'] = pd.DataFrame(adata_merge_filtered.uns[method_name]['pvals'])[str(selected_cluster)].values
    res_pd['pvals_adj'] = pd.DataFrame(adata_merge_filtered.uns[method_name]['pvals_adj'])[str(selected_cluster)].values
    res_pd = res_pd.sort_values(['pvals_adj']) 
    return(res_pd)

# return the filtered gene list dataframe
@st.cache(allow_output_mutation=True)
def marker_filter(res_pd,user_score_min, user_score_max,user_lf_min, user_lf_max):   
    res_pd_filter = res_pd[((res_pd['scores']<user_score_min) | (res_pd['scores']>user_score_max)) 
                           & ((res_pd['logfoldchanges']<user_lf_min) | (res_pd['logfoldchanges']>user_lf_max))]
    res_pd_filter = res_pd_filter.sort_values(['scores','logfoldchanges'],ascending=False)
    res_pd_filter = res_pd_filter.reset_index(drop=True)
    return(res_pd_filter)

res_pd_filter=[]
with tab3:
    st.header('Section 3: Select cluster to perform differentially expressed gene analysis')
    selected_cluster = st.selectbox('Selected cluster',
        tuple([str(leiden_idx) for leiden_idx in list(range(0,leiden_max))]),                   
        (0),
        key="selected_box"
    )
    referece_cluster = st.selectbox('Reference cluster',
        tuple([str(leiden_idx) for leiden_idx in list(range(0,leiden_max)) if str(leiden_idx)!=str(selected_cluster)] + ['rest']),  
        (leiden_max-1),
        key="referece_box"
    )
    fileName = 'c'+str(selected_cluster)+'_vs_'+'c'+str(referece_cluster) 
    data_btn = st.button("Perform wilcoxon analysis")
    
    cluster_name = 'leiden'
    method_name = 'wilcoxon'
    
    #if data_btn == True or (str(selected_cluster) in pd.DataFrame(adata_merge_filtered.uns[method_name]['names']).keys()):
    #    if((str(selected_cluster) in pd.DataFrame(adata_merge_filtered.uns[method_name]['names']).keys())):
    #        print("passed")
    #    else:
    #        compute_DEG(cluster_name,selected_cluster,referece_cluster)
    if data_btn == True:
        compute_DEG(cluster_name,selected_cluster,referece_cluster)
    elif method_name in adata_merge.uns:
        if str(selected_cluster) in pd.DataFrame(adata_merge_filtered.uns[method_name]['names']).keys():
            print("passed")
    else:
        st.write("Please click the \"Perform wilcoxon analysis\" botton for the data processing.")
        st.stop()
    if method_name in adata_merge.uns:
        if str(selected_cluster) in pd.DataFrame(adata_merge_filtered.uns[method_name]['names']).keys():        
            res_pd = get_wilcoxon_result(adata_merge_filtered,selected_cluster)
            
            
            score_max = int(np.floor(max(res_pd['scores'].values)))
            score_min = int(np.floor(min(res_pd['scores'].values)))
            lf_max = int(np.floor(max(res_pd['logfoldchanges'].values)))
            lf_min = int(np.floor(min(res_pd['logfoldchanges'].values)))
            
            st.markdown("Scores and log foldchange cutoff.")
            user_score_min, user_score_max = st.slider('scores', 
                                           max_value=score_max,
                                           min_value=score_min,
                                           value=(-3, 3))
            user_lf_min, user_lf_max = st.slider('log foldchanges', 
                                           max_value=float(lf_max), 
                                           min_value=float(lf_min),
                                           value=(-1.0, 1.0))
            res_pd_filter = marker_filter(res_pd,user_score_min, user_score_max,user_lf_min, user_lf_max)
            
            st.subheader(fileName+" differetially expressed genes sorted by scores (-log(p-value)*Fold Change).")
            st.write(res_pd_filter)
            st.markdown(get_table_download_link(pd.DataFrame(res_pd_filter), fileName = fileName+'_DEG list result'), unsafe_allow_html=True)


#sampleNames=[]
#for i in range(0,len(degs)):
#    sampleNames.append(degs[i][0])
#
#if st.checkbox('Show DEG results table', value=True):
#    SampleNameButton1 = st.radio(
#         "selected sample",
#         sampleNames,key='DEG')
#    if SampleNameButton1 in [i[0] for i in degs]:
#        idx=[i[0] for i in degs].index(SampleNameButton1)
#        deg=degs[idx]
#        sampleName=deg[0]
#        st.write('You selected: '+sampleName)
#        if 'Unnamed: 0' in degs[idx][1].keys():
#            degs[idx][1] = degs[idx][1].drop(['symbol'], axis=1, errors='ignore')
#            degs[idx][1] = degs[idx][1].rename(columns = {"Unnamed: 0":'symbol'}) #, inplace = True
#        
#        st.write(degs[idx][1])
#        st.markdown(get_table_download_link(pd.DataFrame(degs[idx][1]), fileName = degs[idx][0]+' DEG list result'), unsafe_allow_html=True)
        
with tab4:       
    ########
    st.header('Section 4: Perform PAGER Analysis')
    st.markdown("The list of significantly differentially expressed genes (DEG) is then passed to Pathways, Annotated gene lists, and Gene signatures Electronic Repository (PAGER), which offers a network-accessible REST API for performing various gene-set, network, and pathway analyses.")
    
    st.sidebar.subheader('Adjust PAGER Parameters')
    link = 'The PAGER database detail [http://discovery.informatics.uab.edu/PAGER/](http://discovery.informatics.uab.edu/PAGER/)'
    st.sidebar.markdown(link, unsafe_allow_html=True)
    sources = st.sidebar.multiselect('Available Data Sources',
        ("WikiPathway_2021_HUMAN","Reactome_2021","KEGG_2021","Spike","BioCarta","NCI-Nature Curated","GeoMx Cancer Transcriptome Atlas","Microcosm Targets","TargetScan","mirTARbase","NGS Catalog","GTEx","HPA-TCGA","HPA-PathologyAtlas","HPA-GTEx","HPA-FANTOM5","HPA-normRNA","HPA-RNAcon","GOA","I2D","Cell","HPA-CellAtlas","CellMarker","GAD","GWAS Catalog","PheWAS","MSigDB","GeneSigDB","PharmGKB","DSigDB","Genome Data","Protein Lounge","Pfam","Isozyme","HPA-normProtein"),
        ("WikiPathway_2021_HUMAN","Reactome_2021","KEGG_2021","Spike","BioCarta","NCI-Nature Curated")
    )
    
    
    olap = st.sidebar.text_input("Overlap ≥", 1)
    sim = st.sidebar.slider('Similarity score ≥', 0.0, 1.0, 0.05, 0.01)
    fdr = st.sidebar.slider('-log2-based FDR Cutoff', 0, 300, 3, 1)
    fdr = np.power(2,-np.float64(fdr))
    
    # modified PAG enrichment
    PAGERSet=pd.DataFrame()
    deg_names=[]
    pag_ids=[]
    pags=[]
    PAG_val=dict()
    # Remove nan from gene list.
    
    ## simple upper case in transforming homologous gene symbol from mouse to human
    #res_pd_filter['human_symbol'] = [x.upper() for x in res_pd_filter['names'].values.tolist() if str(x) != 'nan']
    # homolog
    homologene = pd.read_csv('homologene'+'/'+'homologene_builld68.data.txt',
                             sep="\t",header=None)
    homocolnames = ["homologene","tax_id","gene_id","symbol","ensembl","protein"]
    homologene.columns = homocolnames
    human_gene = homologene[(homologene['tax_id'] == 9606)]
    mouse_gene = homologene[(homologene['tax_id'] == 10090)]
    human_mouse = human_gene.merge(mouse_gene,left_on='homologene', right_on='homologene', how='outer')
    mouse_human_map = human_mouse[human_mouse['symbol_y'].isin(res_pd_filter['names'])]
    mouse_human_gene = mouse_human_map[['symbol_x','symbol_y']]
    mouse_human_gene = mouse_human_gene.rename(columns={'symbol_x':'human','symbol_y':'mouse_symbol'})
    res_pd_filter = res_pd_filter.merge(mouse_human_gene,left_on="names",right_on="mouse_symbol")
    genes = res_pd_filter['human'][~res_pd_filter['human'].isna()].unique()
    
    
    #st.write(genes)
    #pager_run_state = st.text('Calling PAGER REST API ... ')
    if len(genes) != 0:
        pager_output = run_pager(genes, sources, olap, sim, fdr)
        #pager_run_state.text('Calling PAGER REST API ... done!')
        #st.write(pager_output)
        st.subheader('View/Filter Results')
        # Convert GS_SIZE column from object to integer dtype.
        # See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html.
        pager_output = pager_output.astype({'GS_SIZE': 'int32'})
        gs_sizes = pager_output['GS_SIZE'].tolist()
        # Figure out the min and max GS_SIZE within the PAGER output.
        min_gs_size = min(gs_sizes)
        max_gs_size = max(gs_sizes)
        # Set up a range slider. Cool!
        # See https://streamlit.io/docs/api.html#streamlit.slider.
        user_min, user_max = st.slider('GS_SIZE Range', max_value=max_gs_size, value=(min_gs_size, max_gs_size))
        filtered_output = pager_output[pager_output['GS_SIZE'].between(user_min, user_max)]
        filtered_output = filtered_output.sort_values(['pFDR'],ascending=True)
        filtered_output = filtered_output.reset_index(drop=True)
        st.subheader(fileName+" enriched PAG sorted by p-value False Discovery Rate (FDR).")    
        st.write(filtered_output)            
        if(len(filtered_output.index)>0):
            for row in filtered_output.iloc[:,[0,1,-1]].values:
                pag_id=str(row[0])+"_"+str(row[1])
                pags.append(pag_id)
                pag_ids=pag_ids+[pag_id]
                val=-np.log(row[2])/np.log(10)
                PAG_val[pag_id]=val
        filtered_output['SAMPLE'] = "c"+str(selected_cluster)
        PAGERSet = PAGERSet.append(filtered_output)
        st.markdown(get_table_download_link(filtered_output, fileName = fileName +' geneset enrichment result'), unsafe_allow_html=True)
    PAGERSet = pd.DataFrame(PAGERSet)


##st.write(PAGERSet.shape[1])
#if PAGERSet.shape[1] < 2:
#    st.write("No enriched PAGs found. Try a lower similarity score or a lower -log2-based FDR cutoff and rerun.")
#    st.stop()
##st.write(PAGERSet)
##st.write(pag_ids)
#PAGERSet['PAG_FULL'] = pag_ids
#pag_ids=list(set(pag_ids))
#
#st.write("Select the samples and narrow down the PAGs in enriched those selected samples")
#opts = []
#for deg_name in deg_names:
#    opts.append((deg_name))
#known_variables = {symbol: st.checkbox(f"{symbol}", value = True) for symbol in opts}
#selected_pags = [key for key,val in known_variables.items() if val == True]#
#if(len(selected_pags) == 0):
#    st.write("Please select at least one sample to generate the heatmap.")
#    st.stop()   
#pag_ids=list(set(PAGERSet[PAGERSet['SAMPLE'].isin(selected_pags)]['PAG_FULL'].tolist()))
##st.write(pag_ids)
#mtx=np.zeros((len(pag_ids), len(deg_names)))
#for pag_idx in range(0,len(pag_ids)):
#    for name_idx in range(0,len(deg_names)):
#        if(deg_names[name_idx]+pag_ids[pag_idx] in PAG_val.keys()):
#            mtx[pag_idx,name_idx]=PAG_val[deg_names[name_idx]+pag_ids[pag_idx]]
#
#
#orderExpect = treatment_data.index.tolist()[0:]
#orderIdx = [sampleNames.index(i) for i in orderExpect]
##st.write([len(pag_id) for pag_id in pag_ids])
#
#width_ratio_heatmap = st.slider('Width ratio of heatmap (increase to widen the heatmap)', 0.1, 5.0, 1.0, 0.1)
#
#### heatmap ###
#heatmapBtn = st.button("Generate the heatmap")
#if heatmapBtn == True:
#    plt = generateheatmap(np.array(mtx)[::,orderIdx]
#                              ,np.array(deg_names)[orderIdx]
#                              ,pag_ids
#                              ,rowCluster=True
#                              ,colCluster = False
#                              ,width_ratio=width_ratio_heatmap)
#    st.pyplot(plt)

#st.header('Section 4 out of 5: Generate the heatmap of the samples\' DEG enrichment result (' + str(len(pag_ids)) + ' PAGs)')
#from PIL import Image
#image = Image.open('./heatmap.png')
#st.image(image, caption='Sample-PAG association',
#         use_column_width=True)
#st.pyplot(caption='Sample-PAG association')

### generate PPI data ###
@st.cache(allow_output_mutation=True)
def PPIgeneration(geneInt,symbol2idx):      
    idxPair=[]
    PPI=[]
    for pair in geneInt['data']:
        idxPair.append((symbol2idx[pair['SYM_A']],symbol2idx[pair['SYM_B']]))
        PPI.append((pair['SYM_A'],pair['SYM_B']))
        
    return(idxPair,PPI,idx2symbol)

with tab5:    
    st.header('Section 5: Generate the network of the selected PAG')
    st.write('Select a PAG_ID here:')
    #st.write(pag_ids)
    PAGid = st.selectbox(
        'Available PAG_IDs',
        tuple(pag_ids),
        key = "PAG_ID_box"
        )
    #st.write(PAGid)
    
    
    ID_only = re.sub("([A-Z0-9]+)_[^_]*","\\1",str(PAGid))
    link = "For the selected PAG "+ str(PAGid)+"'s gene information. (http://discovery.informatics.uab.edu/PAGER/index.php/geneset/view/"+ID_only+")"
    st.markdown(link, unsafe_allow_html=True)
    
    
    geneInt = run_pager_int(ID_only)
    geneRanked = pag_ranked_gene(ID_only)
    #st.write(geneRanked)
    
    idx2symbol = dict()
    symbol2idx = dict()
    symbol2size = dict()
    idx=0
    geneRanked['RP_SCORE'].fillna(0.1, inplace=True)
    geneRanked['RP_SCORE'] = geneRanked['RP_SCORE'].astype(float)
    geneRanked['node_size'] = geneRanked['RP_SCORE'] *4
    st.write(geneRanked)
    for gene_idx in range(0,geneRanked.shape[0]):
        gene = geneRanked.iloc[gene_idx,]
        #st.write(gene)
        symbol2idx[gene['GENE_SYM']] = str(idx)
        #st.write(gene['RP_SCORE'])
        #symbol2size[gene['GENE_SYM']] = gene['RP_SCORE']
        if(gene['RP_SCORE'] is not None):
            symbol2size[gene['GENE_SYM']] = gene['node_size']
        else:
            symbol2size[gene['GENE_SYM']] = 1
        idx2symbol[str(idx)] = gene['GENE_SYM']
        idx+=1
    
    
    (idxPair,PPI,idx2symbol) = PPIgeneration(geneInt,symbol2idx)
    
    
    #st.write(PPI)
    
    # spring force layout in networkx
    #import networkx as nx
    #G=nx.Graph()
    #G.add_nodes_from(idx2symbol.values())
    #G.add_edges_from(PPI)
    #pos=run_force_layout(G)
    
    
    
    #SampleNameButton = st.radio(
    #     "selected sample",
    #     sampleNames,key='network')
    colorMap = dict()
    
    #if SampleNameButton in [i[0] for i in degs]:    
        #idx=[i[0] for i in degs].index(SampleNameButton)
    #for idx in orderIdx:     
        #deg=degs[idx]
        #sampleName=deg[0]
    
    config = Config(height=500, width=700, nodeHighlightBehavior=True, highlightColor="#F7A7A6", directed=False,
          collapsible=True,              
          node={'labelProperty':'label',"strokeColor": "black"},
          #, link={'labelProperty': 'label', 'renderLabel': True}
          link={'color': "#d3d3d3"},
                    key="agraph_"+fileName
       )
    st.write("Sample:"+fileName)
    #deg_results=deg[1]
    deg_results = res_pd_filter
    genesExp = [x for x in deg_results[['human','logfoldchanges']].values.tolist()] # if str(x[0]) != 'nan'
    #st.write(np.array(genesExp)[:,-1])
    # expression data in network
    expInNetwork=np.array(genesExp)[np.logical_or.reduce([np.array(genesExp)[:,0] == x for x in idx2symbol.values()])].tolist()
    
    # show expression table
    st.write("Gene expression within the selected PAG:")
    expInNetworkArr = np.array(expInNetwork)
    expInNetworkArrSorted = np.array(sorted(expInNetworkArr,key = lambda expInNetworkArr:np.float64(expInNetworkArr[1]), reverse=True))
    DataE=pd.DataFrame(expInNetworkArrSorted)
    DataE.rename(columns={0:'symbol',1:'log2FC'},inplace=True)
    st.write(DataE)
    
    ### show expression figure ###
    marker_genes = res_pd_filter[res_pd_filter.human.isin(DataE.symbol.values)].mouse_symbol
    option = st.selectbox(
         'Type of gene expression\'s figure',
         ('dotplot', 'violin','matrix'))
    st.write('You selected:', option)
    sc.tl.dendrogram(adata_merge_filtered,groupby="leiden")
    if option == 'dotplot':      
        dp = sc.pl.dotplot(adata_merge_filtered, marker_genes.values, groupby='leiden', return_fig=True,categories_order=sel_cluster)
        st.pyplot(dp.add_totals().style(dot_edge_color='black', dot_edge_lw=0.5, cmap='viridis').show())
    elif option == 'heatmap':
        sc.tl.dendrogram(adata_merge_filtered,groupby="leiden")
        st.pyplot(sc.pl.heatmap(adata_merge_filtered, marker_genes.values, groupby='leiden', swap_axes=True,cmap="viridis"))
    elif option == 'violin':
        st.pyplot(sc.pl.stacked_violin(adata_merge_filtered, marker_genes.values, groupby='leiden',cmap="viridis",categories_order=sel_cluster))
    elif option == 'matrix':   
        st.pyplot(sc.pl.matrixplot(adata_merge_filtered, marker_genes.values, groupby='leiden',cmap="viridis",categories_order=sel_cluster))
    
    
    
    if np.size(np.array(expInNetwork))>0:
        zeroInNetwork=[[i,'0'] for i in idx2symbol.values() if i not in np.array(expInNetwork)[:,0]]
    else:
        zeroInNetwork=[[i,'0'] for i in idx2symbol.values()]
    for i in zeroInNetwork:
        expInNetwork.append(i)
        
    
    
    # And a data frame with characteristics for your nodes in networkx
    carac = pd.DataFrame({'ID':np.array(expInNetwork)[:,0], 
                          'myvalue':[np.float64(i) for i in np.array(expInNetwork)[:,1]] })
    
    # Plot it, providing a continuous color scale with cmap:
    # Here is the tricky part: I need to reorder carac, to assign the good color to each node
    carac = carac.set_index('ID')
    #carac = carac.reindex(G.nodes())
    # load network function 
    #X = Network.generateNetwork(carac,pos,PPI)
    #st.pyplot(plt,caption=sampleName)
    #image = Image.open('./network.png')
    #st.image(image, caption=sampleName,
    #     use_column_width=True)
    #st.write(X.nodes)
    #st.write(X.edges)
    #st.write(carac.to_dict()["myvalue"])
    
    #st.write(newcmp)
    st.subheader(fileName+" enriched "+str(PAGid)+"'s gene network")  
    max_val = max([np.abs(val) for val in carac.to_dict()["myvalue"].values()])
    
    #st.write(max_val)
    if max_val != float(0):
        nodes = [] 
        for i in idx2symbol.values():             
            #carac.to_dict()["myvalue"][str(i)]
            nodes.append(Node(
                id=i, 
                label=str(i), 
                size=symbol2size[str(i)],#400,                            
                color=hex_map[int( carac.to_dict()["myvalue"][i]/max_val*colorUnit)+colorUnit]
            )
            ) # includes **kwargs
        #edges = [Edge(source=i, label="int", target=j,color="#d3d3d3") for (i,j) in X.edges] # includes **kwargs  type="CURVE_SMOOTH"
        edges = [Edge(source=pair[0], label="int", target=pair[1],color="#d3d3d3") for pair in PPI]
        
        return_value = agraph(nodes=nodes, 
                      edges=edges, 
                      config=config)
        #agraph(list(idx2symbol.values()), (PPI), config)
        st.markdown(get_table_download_link(pd.DataFrame(PPI), fileName = ' '+fileName+' '+str(PAGid)+' data for interactions'), unsafe_allow_html=True)
        st.markdown(get_table_download_link(pd.DataFrame(DataE), fileName = ' '+fileName+' '+str(PAGid)+' data for gene expressions'), unsafe_allow_html=True)
    else:
        st.write("No expression.")
    #except:
    #    st.write("You select nothing.")




    st.header('Cite:')
    st.write("PAGER analysis:")
    st.write("Zongliang Yue, Qi Zheng, Michael T Neylon, Minjae Yoo, Jimin Shin, Zhiying Zhao, Aik Choon Tan, Jake Y Chen, PAGER 2.0: an update to the pathway, annotated-list and gene-signature electronic repository for Human Network Biology, Nucleic Acids Research, Volume 46, Issue D1, 4 January 2018, Pages D668–D676,https://doi.org/10.1093/nar/gkx1040")
    st.markdown("http://discovery.informatics.uab.edu/PAGER/")
    st.write("Protein-Protein Interactions (PPIs) in network construction:")
    st.write("Jake Y. Chen, Ragini Pandey, and Thanh M. Nguyen, (2017) HAPPI-2: a Comprehensive and High-quality Map of Human Annotated and Predicted Protein Interactions, BMC Genomics volume 18, Article number: 182")
    st.markdown("http://discovery.informatics.uab.edu/HAPPI/")        
    
    st.header('About us:')
    st.write(f"If you have questions or comments about the database contents, please email Dr. Jake Chen, jakechen@uab.edu.")
    st.write("If you need any technical support, please email Zongliang Yue, zongyue@uab.edu.")
    st.write("Our lab: AI.MED Laboratory, University of Alabama at Birmingham, Alabama, USA. Link: http://bio.informatics.uab.edu/")
##for idx in range(0,len(degs)):
##    deg=degs[idx]
##    sampleName=deg[0]
##    deg_results=deg[1]
##    genesExp = [x for x in deg_results[['symbol','log2FoldChange']].values.tolist() if str(x[0]) != 'nan']
##    # expression data in network
##    expInNetwork=np.array(genesExp)[np.logical_or.reduce([np.array(genesExp)[:,0] == x for x in idx2symbol.values()])].tolist()
##    if np.size(np.array(expInNetwork))>0:
##        zeroInNetwork=[[i,'0'] for i in idx2symbol.values() if i not in np.array(expInNetwork)[:,0]]
##    else:
##        zeroInNetwork=[[i,'0'] for i in idx2symbol.values()]
##    for i in zeroInNetwork:
##        expInNetwork.append(i)
##    # And a data frame with characteristics for your nodes
##    carac = pd.DataFrame({ 'ID':np.array(expInNetwork)[:,0], 'myvalue':[np.float(i) for i in np.array(expInNetwork)[:,1]] })
##    # Plot it, providing a continuous color scale with cmap:
##    # Here is the tricky part: I need to reorder carac, to assign the good color to each node
##    carac= carac.set_index('ID')
##    carac=carac.reindex(G.nodes())
##
##    plt=Network.generateNetwork(carac,pos,PPI)
##    st.pyplot(caption=sampleName)

##plt.savefig("./network.png", dpi=100,transparent = True, bbox_inches = 'tight', pad_inches = 0)
##
##from PIL import Image
##image = Image.open('./network.png')
##st.image(image, #caption='Sample-PAG association',
##         use_column_width=True)