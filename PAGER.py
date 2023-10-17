############################################
### PAGER functions for PAGER server API ###
############################################
import requests
import numpy as np
import pandas as pd
class PAGER():
    def __init__(self):
        self.params = {}
        
    ######   
    # run_pager is a function connected to PAGER api to perform hypergeometric test and retrieve enriched PAGs associated to a list of genes
    # The input parameters are:
    # 1.genes: a list of gene symbols.
    # 2.source: a list of sources refered to 'http://discovery.informatics.uab.edu/PAGER/index.php/pages/help'
    # 3.Type: a list of PAG types consisting of 'P', 'A' or 'G'
    # 4.minSize: the allowed minimum size of PAG genes
    # 5.maxSize: the allowed maximum size of PAG genes
    # 6.similarity: the similarity score cutoff [0,1]
    # 7.overlap: the allowed minimum overlap genes
    # 8.nCoCo: the minimum nCoCo score
    # 9.pValue: p-value cutoff
    # 10.FDR: false discovery rate 
    ######
    def run_pager(self, genes,**kwargs):       
        source = kwargs['source'] if 'source' in kwargs.keys() else ['KEGG_2021_HUMAN', 'WikiPathway_2021', 'BioCarta', 'Reactome_2021', 'Spike']
        Type = kwargs['Type'] if 'Type' in kwargs.keys() else 'All'
        minSize = kwargs['minSize'] if 'minSize' in kwargs.keys() else 1
        maxSize = kwargs['maxSize'] if 'maxSize' in kwargs.keys() else 2000
        sim = kwargs['similarity'] if 'similarity' in kwargs.keys() else 0
        overlap = kwargs['overlap'] if 'overlap' in kwargs.keys() else '1'
        organism = kwargs['organism'] if 'organism' in kwargs.keys() else 'All'
        nCoCo = kwargs['nCoCo'] if 'nCoCo' in kwargs.keys() else '0'
        pvalue = kwargs['pvalue'] if 'pvalue' in kwargs.keys() else 0.05
        FDR = kwargs['FDR'] if 'FDR' in kwargs.keys() else 0.05
        # Set up the call parameters as a dict.
        params = {}
        # Work around PAGER API form encode issue.
        params['genes'] = '%20'.join(genes)
        params['source'] = '%20'.join(source)
        params['type'] = Type
        params['ge'] = minSize
        params['le'] = maxSize        
        params['sim'] = str(sim)
        params['olap'] = str(overlap)
        params['organism'] = organism
        params['cohesion'] = str(nCoCo)
        params['pvalue'] = pvalue
        params['FDR'] = FDR
        response = requests.post('http://discovery.informatics.uab.edu/PAGER/index.php/geneset/pagerapi', data=params)
        #print(response.request.body)
        return pd.DataFrame(response.json())
    
    # pathMember is a function connected to PAGER api to retrieve the membership of PAGs using a list of PAG IDs
    def pathMember(self, PAG_IDs):
        # Set up the call parameters as a dict.
        params = {}
        params['pag'] = ','.join(PAG_IDs)
        # Work around PAGER API form encode issue.
        response = requests.post('http://discovery.informatics.uab.edu/PAGER/index.php/geneset/get_members_by_ids/', data=params)
        #print(response.request.body)
        return pd.DataFrame(response.json()['data'])       
    
    # pathInt is a function connected to PAGER api to retrieve the m-type relationships of PAGs using a list of PAG IDs 
    def pathInt(self, PAG_IDs):
        # Set up the call parameters as a dict.
        params = {}
        params['pag'] = ','.join(PAG_IDs)
        # Work around PAGER API form encode issue.
        response = requests.post('http://discovery.informatics.uab.edu/PAGER/index.php/pag_pag/inter_network_int_api/', data=params)
        #print(response.request.body)
        return pd.DataFrame(response.json()['data'])
    
    # pathReg is a function connected to PAGER api to retrieve the r-type relationships of PAGs using a list of PAG IDs   
    def pathReg(self, PAG_IDs):
        # Set up the call parameters as a dict.
        params = {}
        params['pag'] = ','.join(PAG_IDs)
        # Work around PAGER API form encode issue.
        response = requests.post('http://discovery.informatics.uab.edu/PAGER/index.php/pag_pag/inter_network_reg_api/', data=params)
        #print(response.request.body)
        return pd.DataFrame(response.json()['data'])
    
    # pagRankedGene is a function connected to PAGER api to retrieve RP-ranked genes with RP-score of the given PAG_IDs
    def pagRankedGene(self, PAGid):
        response = requests.get('http://discovery.informatics.uab.edu/PAGER/index.php/genesinPAG/viewgenes/'+PAGid)
        return pd.DataFrame(response.json()['gene'])
    
    # pagGeneInt is a function connected to PAGER api to retrieve gene interaction network
    def pagGeneInt(self, PAGid):
        response = requests.get('http://discovery.informatics.uab.edu/PAGER/index.php/pag_mol_mol_map/interactions/'+PAGid)
        return pd.DataFrame(response.json()['data'])
    
    # pagGeneReg is a function connected to PAGER api to retrieve gene regulatory network
    def pagGeneReg(self, PAGid):
        response = requests.get('http://discovery.informatics.uab.edu/PAGER/index.php/pag_mol_mol_map/regulations/'+PAGid)
        return pd.DataFrame(response.json()['data'])
    
    # path_NGSEA is a function connected to PAGER api to generate the network-based GSEA result
    import requests
    def path_NGSEA(SELF, genes, PAGmember):
        geneExpStr = ''
        for rowIdx in range(0,genes.shape[0]):
            geneExpStr = geneExpStr + genes.iloc[rowIdx,0]+'\\t\\t'+genes.iloc[rowIdx,1] + "\\t\\t\\t"
        PAGsetsStr = ''
        for rowIdx in range(0,PAGmember.shape[0]):
            PAGsetsStr = PAGsetsStr + PAGmember.iloc[rowIdx,0]+'\\t\\t'+PAGmember.iloc[rowIdx,1] + "\\t\\t\\t"
        params = {}
        params['geneExpStr'] = geneExpStr
        params['PAGsetsStr'] = PAGsetsStr
        response = requests.post('http://discovery.informatics.uab.edu/PAGER/index.php/geneset/ngseaapi/', data=params)
        #print(response.request.body)
        return pd.DataFrame(response.json()['data'])
