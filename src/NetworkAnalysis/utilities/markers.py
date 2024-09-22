
import pandas as pd 

import pickle 

def createMarkerGenes(genesList):
    df = pd.DataFrame(data=genesList, columns=["Gene"])
    df = addTcga(df)
    df = addVU(df)
    df = addLund(df)
    df = addTF(df)
    df.set_index("Gene", inplace=True)
    return df

def addTcga(df, gene_col = "Gene"):
    """
    Receives a DataFrame with the indexes the genes and label column set to "label".
    Populates the genes specific to TCGA

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """

    luminal_markers = ["KRT20", "PPARG", "FOXA1", "GATA3", "SNX31", "UPK1A", "UPK2", "FGFR3"]

    basal_markers = ["CD44", "KRT6A", "KRT5", "KRT14", "COL17A1"]

    squamos_markers = ["DSC3", "GSDMC", "TCGM1", "PI3", "TP63"]

    immune_markers = ["CD274", "PDCD1LG2", "IDO1", "CXCL11", "L1CAM", "SAA1"]

    neural_diff = ["MSI1", "PLEKHG4B", "GNG4", "PEG10", "RND2", "APLP1", "SOX2", "TUBB2B"]
    # TCGA
    df.loc[df[gene_col].isin(luminal_markers), "TCGA"] = "Luminal"
    df.loc[df[gene_col].isin(basal_markers), "TCGA"] = "Basal"
    df.loc[df[gene_col].isin(squamos_markers), "TCGA"] = "Squamous"
    df.loc[df[gene_col].isin(immune_markers), "TCGA"] = "Immune"
    df.loc[df[gene_col].isin(neural_diff), "TCGA"] = "Ne"

    return df

def addVU(df, gene_col="Gene"):

    results_path = "/Users/vlad/Documents/Code/York/iNet/data/gene_contrib/"
    # 100 genes
    # with open(results_path + 'uniq_genes.pickle', 'rb') as handle:
    #     unique_genes = pickle.load(handle)
    # for key, val in unique_genes.items():
    #     df.loc[df[gene_col].isin(val), "VU"] = key

    # # 200
    # with open(results_path + 'uniq_genes_200.pickle', 'rb') as handle:
    #     unique_genes = pickle.load(handle)
    # for key, val in unique_genes.items():
    #     df.loc[df[gene_col].isin(val), "VU_200"] = key

    # # 500
    # with open(results_path + 'uniq_genes_500.pickle', 'rb') as handle:
    #     unique_genes = pickle.load(handle)
    # for key, val in unique_genes.items():
    #     df.loc[df[gene_col].isin(val), "VU_500"] = key

    # # 200 relative
    # with open(results_path + 'uniq_genes_rel_200.pickle', 'rb') as handle:
    #     unique_genes = pickle.load(handle)
    # for key, val in unique_genes.items():
    #     df.loc[df[gene_col].isin(val), "VU_rel_200"] = key

    # 500 relative
    with open(results_path + 'uniq_genes_rel_500.pickle', 'rb') as handle:
        unique_genes = pickle.load(handle)
    for key, val in unique_genes.items():
        df.loc[df[gene_col].isin(val), "VU_rel_500"] = key

    # 500 relative gc42
    with open(results_path + 'uniq_genes_rel_500_gc42.pickle', 'rb') as handle:
        unique_genes = pickle.load(handle)
    for key, val in unique_genes.items():
        df.loc[df[gene_col].isin(val), "VU_rel_500_gc42"] = key

    # 200 relative gc42
    with open(results_path + 'uniq_genes_rel_200_gc42.pickle', 'rb') as handle:
        unique_genes = pickle.load(handle)
    for key, val in unique_genes.items():
        df.loc[df[gene_col].isin(val), "VU_rel_200_gc42"] = key

    # all genes relative gc42
    # with open(results_path + 'all_genes_rel_200_gc42.pickle', 'rb') as handle:
    #     unique_genes = pickle.load(handle)
    # for key, val in unique_genes.items():
    #     df.loc[df[gene_col].isin(val), "VU_all_200_gc42"] = key

    with open(results_path + 'uniq_genes_rel_100_gc42.pickle', 'rb') as handle:
        unique_genes = pickle.load(handle)
    for key, val in unique_genes.items():
        df.loc[df[gene_col].isin(val), "VU_rel_100_gc42"] = key

    with open(results_path + 'uniq_genes_rel_50_gc42.pickle', 'rb') as handle:
        unique_genes = pickle.load(handle)
    for key, val in unique_genes.items():
        df.loc[df[gene_col].isin(val), "VU_rel_50_gc42"] = key


    return df

def addTF(df, gene_col="Gene"):

    results_path = "/Users/vlad/Documents/Code/York/iNet/data/"
    tf_list = list(pd.read_csv(results_path + "TF_names_v_1.01.txt")["TF"])

    df["TF"] = 0
    df.loc[df[gene_col].isin(tf_list), "TF"] = 1
    
    return df


# lund
def addLund(df, gene_col = "Gene"):
    # both ba_sq_inf and mes_like
    lund_qtc1 = ["FLI1", "FOXP3", "ILKZF1", "IRF4", "IRF8", "RUNX3", "SCML4", "SPI1", "STAT4", "TBX21", "TFEC"]
    lund_qtc2 = ["AEBP1", "BNC2", "GLI2", "GLIS1", "HIC1", "MSC", "PPRX1", "PPRX2", "TGFB1I1", "TWIST1"]
    lund_qtc3 = ["EBF1", "HEYL", "LEF1", "MEF2C", "TCF4", "ZEB1", "ZEB2"]
    lund_qtc8 = ["GATA5", "HAND1", "HAND2", "KLF16"]
    lund_qtc17 = ["ARID5A", "BATF3", "VENTX"]
    lund_ba_mes = lund_qtc1 + lund_qtc2 + lund_qtc3 + lund_qtc8 + lund_qtc17

    lund_ba_sq = ["BRIP1", "E2F7", "FOXM1", "ZNF367", "IRF1", "SP110", "STAT1"]
    lund_mes = ["TP53", "RB1", "FGFR3", "ANKHD1", "VIM", "ZEB2"]
    ba_sq_inf = ["CDH3", "EGFR"]

    lund_sc_ne = ["CHGA", "SYP", "ENO2", "EPCAM"] #Highly expressed

    df.loc[df[gene_col].isin(lund_qtc1), "Lund"] = "QTC1"
    df.loc[df[gene_col].isin(lund_qtc2), "Lund"] = "QTC2"
    df.loc[df[gene_col].isin(lund_qtc3), "Lund"] = "QTC3"
    df.loc[df[gene_col].isin(lund_qtc8), "Lund"] = "QTC8"
    df.loc[df[gene_col].isin(lund_qtc17), "Lund"] = "QTC17"

    df.loc[df[gene_col].isin(lund_ba_mes), "Lund"] = "Ba/Mes"
    df.loc[df[gene_col].isin(lund_ba_sq), "Lund"] = "Ba/Sq"
    df.loc[df[gene_col].isin(lund_mes), "Lund"] = "Mes"
    df.loc[df[gene_col].isin(ba_sq_inf), "Lund"] = "Ba/Sq/Inf"
    df.loc[df[gene_col].isin(lund_sc_ne), "Lund"] = "Sc/Ne"

    return df
    
# immune cells
def addImune(custom_traces):
    b_cells = ["BCL2", "BCL6", "CD19", "CD1D", "CD22", "CD24", "CD27", "CD274","CD34", "CD38", "CD40","CD44","CD5","CD53","CD69","CD72", "CD79A", "CD79B", "CD80", "CD86", "CD93", "CR2", "CXCR4", 'CXCR5',"FAS","FCER2", "FCRL4" "HAVCR1","IL10", 'IL2RA','IL7R','IRF4','ITGAX', 'LILRB1','MME','MS4A1','NT5E','PDCD1LG2','PRDM1','PTPRC','SDC1','SPN','TFRC','TLR9','TNFRSF13B','TNFRSF13C','TNFRSF17','XBP1']
    t_cells = ['CD4', 'CD8', 'CCR4', 'CCR5', 'CCR6', 'CCR7', 'CCR10', 'CD127', 'CD27', 'CD28', 'CD38', 'CD58', 'CD69', 'CTLA4', 'CXCR3', 'FAS', 'IL2RA',
            'IL2RB', 'ITGAE', 'ITGAL', 'KLRB1', 'NCAM1', 'PECAM1', 'PTGDR2', 'SELL', 'IFNG', 'IL10', 'IL13', 'IL17A', 'IL2', 'IL21','IL22', 'IL25', 'IL26', 'IL4', 'IL5', 'IL9', 'TGFB1', 'TNF', 'AHR', 'EOMES','FOXO4', 'FOXP1', 'FOXP3', 'GATA3','IRF4', 'LEF1', 'PRDM1', 'RORC','STAT4', 'TBX21','TCF7', 'GZMA']

    nk_cells = ['B3GAT1','CCR7','CD16','CD2','CD226','CD244','CD27','CD300A','CD34','CD58','CD59','CD69','CSF2','CX3CR1','CXCR1','CXCR3','CXCR4','EOMES','GZMB','ICAM1','IFNG','IL1R1','IL22','IL2RB','IL7R','ITGA1','Itga2','ITGAL','ITGAM','ITGB2','KIR2DL1','KIR2DL2','KIT','Klrb1c','KLRC1','KLRC2','KLRD1','KLRF1','KLRG1','KLRK1','LILRB1','Klra4','Klra8','NCAM1','NCR1','NCR2','NCR3','PRF1','SELL','SIGLEC7','SLAMF6','SPN','TBX21','TNF']

    macrophages_cells = [ 'ADGRE1','CCR2','CD14','CD68','CSF1R','Ly6c1','MARCO','MRC1','NOS2','PPARG','SIGLEC1','TLR2','ARG1','CD163','CD200R1','CD80','CD86','CLEC10A','CLEC7A','CSF2','CX3CR1','FCGR1A','ITGAM','MERTK','PDCD1LG2','Retnla','TNF','CCL22','CD36','CD40','IL10','IL1B','IL6','LGALS3','TLR4','CCL2','CCR5','CD209','CD63','CD86','CSF1','CXCL2','FCGR3A','IFNG','IL4','IRF4','ITGAX','MSR1','PDGFB','PTPRC','STAT6','TIMD4','Chil3','CLEC6A','IL1R1','ITGB2','PDCD1LG2','TLR7']

    monocyte_cells = ['CD14','CD16','CSF1R','CX3CR1','ITGAM','ITGAX','LY6C1','CCR2','CXCR4','FCGR1A','SELL','SPN','ADGRE1','CCR7','TNF','CD86','IL10','IL1B','MERTK','TREML4','CD209','NR4A1','Ly6a','PTPRC','IL3RA','CD27','CCR5','CD32','CD1A','MRC1','ITGB3','CD9','CXCR6','CCR1','FLT3','KLF2','CLEC12A','CCR6','CCR8','CD68','CLEC7A','KIT','MAF','MAFB','SPI1','CD1C','PPARG','CEBPB','ITGAE','TEK']

    custom_traces.append({"genes":b_cells, "title": "B markers"})
    custom_traces.append({"genes":t_cells, "title": "T markers"})
    custom_traces.append({"genes":nk_cells, "title": "NK markers"})
    custom_traces.append({"genes":macrophages_cells, "title": "Macrophae markers"})
    custom_traces.append({"genes":monocyte_cells, "title": "Monocitye markers"})

    return custom_traces

def addIg(custom_traces):
    interferon_g = ["IGHG1","IGHG3","IGHJ2","IGHJ3","IGHJ4","IGHM","IGHV1-69D","IGHV2-70","IGHV3-11","IGHV3-15","IGHV4-39","IGKC","IGKJ1","IGKJ4","IGKV1-5","IGKV1-6","IGKV1-9","IGKV1D-39","IGKV2-28","IGKV4-1","IGLC1","IGLC2","IGLC3","IGLJ2","IGLJ3","IGLV1-40","IGLV1-44","IGLV3-19"]

    sig_gens_int_h = ["CMKLR1","CYTH4","FAM78A","GIMAP1","GIMAP5","IFFO1","IL10RA","LILRB4","NRROS","PLEKHO1","RASSF4","SLAMF8","SLC7A7","TCN2","TFEC"]

    custom_traces.append({"genes":interferon_g, "title": "IG family "})
    custom_traces.append({"genes":sig_gens_int_h, "title": "Significant in Basal 2"})

    return custom_traces

def addBasalSpecific(custom_traces):

    serp_genes = ['SERPINB13','SERPINB3', 'SERPINB4']
    spr_genes = ['SPRR1A', 'SPRR1B', 'SPRR2A', 'SPRR2D', 'SPRR3']

    # cell diff
    s100_genes = ['S100A2', 'S100A7', 'S100A8', 'S100A9']
    cell_diff = ["FGFBP1", 'HSPB1']
    keratines = ['KRT14', 'KRT17', 'KRT5','KRT6C']

    custom_traces.append({"genes":serp_genes, "title": "SERP family"})
    custom_traces.append({"genes":spr_genes, "title": "SPR family"})

    custom_traces.append({"genes":s100_genes, "title": "S100 family"})
    custom_traces.append({"genes":cell_diff + keratines, "title": "Cell diff + keratines"})

    high_exprs_genes_in_large_basal_vlad = ["AKR1C3","AQP3","ELF3","FABP5","GJB2","GJB5","GJB6","KLF5","KRT15","KRT16","KRT6B","LCN2","LYPD3","NDUFA4L2","NECTIN1","NECTIN4","PKP1","PROM2","PTHLH","RAB38","RHOV","S100A14","SEMA4B","SERPINB5","SLPI","THBD","TP63"]
    custom_traces.append({"genes":high_exprs_genes_in_large_basal_vlad, "title": "High expressed in basal large"})

    return custom_traces

# LumInf vs Mixed 
def addLumInfMixed(custom_traces):

    #highly expressed in LumInf
    lumInf_top_branch = ["AEBP1","C1R","C1S","C1orf162","C3","CCN2","COL6A3","COX7A1","CPA3","CTSK","DCN","DPT","EVI2A","FGF7","GAS7","GGT5","GLT8D2","GPR68","GREM1","HSPB1","IGFBP4","IRAG1","ISLR","LUM","LY96","MFAP4","MMP2","MRGPRF","NNMT","PDPN","PODN","RARRES2","RNASE6","SELPLG","SERPING1","TAGLN","THBS1","TIMP3","TMEM119","TMEM273","TPSAB1","TPSB2"]

    lumInf_middle_branch = ["ACTA2","ACTG2","ANXA8","ANXA8L1","C19orf33","C4A","C4B","CCL19","CCL21","CLIC3","CLU","CNN1","COL10A1","COL3A1","COMP","DES","HSPB6","IGHA1","IGHA2","IGHG1","IGHG2","IGHG3","IGHG4","IGHJ2","IGHJ3","IGHJ4","IGHJ5","IGHM","IGHV1-2","IGHV1-3","IGHV1-46","IGHV1-69D","IGHV3-11","IGHV3-15","IGHV3-21","IGHV3-23","IGHV3-30","IGHV3-33","IGHV3-43","IGHV3-48","IGHV3-49","IGHV3-74","IGHV4-34","IGHV4-39","IGHV4-61","IGHV5-51","IGKC","IGKJ1","IGKJ2","IGKJ5","IGKV1-12","IGKV1-33","IGKV1-5","IGKV1-6","IGKV1-9","IGKV1D-39","IGKV2-30","IGKV2D-29","IGKV3-11","IGKV3-15","IGKV3-20","IGKV4-1","IGLC1","IGLC2","IGLC3","IGLJ2","IGLJ3","IGLL5","IGLV1-44","IGLV1-47","IGLV1-51","IGLV2-11","IGLV2-14","IGLV2-23","IGLV2-8","IGLV3-1","IGLV3-19","IGLV3-21","IGLV3-25","IGLV3-9","IGLV4-69","IGLV6-57","JCHAIN","KRT17","LMOD1","LYPD3","MYH11","MYL9","PDLIM3","POSTN","PTGDS","S100P","SFRP2","SFRP4","SNCG","THBS2","TMEM40","TSPAN1"]

    lumInf_bottomn_branch = ["AC132217.4","ACKR1","ACTC1","ADIRF","AKR1C2","C10orf99","CAPNS2","CCL14","CST6","CTB-102L5.9","CXCL17","CYP24A1","DHRS2","FXYD3","HSPB8","IGF2","IGHJ1","IGHV1-18","IGHV1-24","IGHV1-69","IGHV2-5","IGHV3-20","IGHV3-7","IGHV4-31","IGHV4-4","IGHV4-59","IGHV6-1","IGKJ3","IGKJ4","IGKV1-16","IGKV1-17","IGKV1-27","IGKV2-24","IGKV2-28","IGKV3D-15","IGLV1-40","IGLV3-10","IGLV7-46","IGLV8-61","IVL","KRT13","KRT23","KRT7","LY6D","MAL","MMP11","NCCRP1","PADI3","PCP4","PLA2G2A","PSCA","S100A2","S100A9","SYT8","TMEM265","UPK1A","UPK1B","UPK2","UPK3A","UPK3B"]

    lumInf_significant_scatter_plot = ["DES","IGKV1-33","IGLV3-21","KRT23","LY6D","SFRP2","UPK1B"]

    mixed = ["ASRGL1","AURKB","B4GALNT4","B4GALT6","BUB1B","C12orf75","CBX2","CCNE2","CDC25C","CDKAL1","CDKN2A","CENPA","CENPF","CNIH2","COCH","CRABP1","CTD-3193O13.11","DEPDC1","DLGAP5","DLX5","E2F1","E2F3","E2F5","ENHO","ESPL1","EZH2","FAM111B","FOXM1","FSD1","GAL","GGH","GPC2","GTSE1","HES6","HMGB2","HOXB9","KIF15","KIF18B","KIF1A","LINC00668","LMNB1","MELK","MEX3A","MSI1","MT3","MYBL2","MYEF2","NDC80","NEK2","NUF2","NUSAP1","ODC1","PDXP","PIMREG","PLEKHG4B","PLK1","PRAME","RAC3","RAD51AP1","RCOR2","RP11-613M10.6","SBK1","SEPTIN3","SLC29A4","SOX2","STMN1","TMSB15A","TPX2","TUBB2B","UGT8","YBX2","YPEL1"]

    custom_traces.append({"genes": lumInf_top_branch, "title": "LumInf/NS top branch"})
    custom_traces.append({"genes": lumInf_middle_branch, "title": "LumInf/NS middle branch"})
    custom_traces.append({"genes": lumInf_bottomn_branch, "title": "LumInf/NS bottom branch"})
    custom_traces.append({"genes": lumInf_significant_scatter_plot, "title": "LumInf/NS significant scatter"})

    custom_traces.append({"genes": mixed, "title": "Mixed"})

    return custom_traces

def addSigGenes(custom_traces):

    small = ['IGKJ1', 'IGKV3-20', 'IGLC2', 'IGKV2-28', 'IGKJ2', 'IGHJ4', 'IGLC1', 'IGKV4-1', 'H19', 'CD14', 'IGHG1', 'C1QB', 'IGHJ5', 'POSTN', 'HTRA3', 'C1QC', 'IGKV1D-39', 'CTHRC1', 'IGHG2', 'AEBP1', 'IGHG3', 'HLA-DQA1', 'IGLC3', 'NNMT', 'CCN2', 'IGKJ4', 'MMP11', 'CTSK']

    # large = ['KRT5', 'DSP', 'S100A2', 'H19', 'LY6D', 'CDH3', 'AQP3', 'MALL', 'KRT16', 'S100A8', 'SERPINB5', 'ANXA8L1', 'SLPI', 'COL17A1', 'KRT14', 'TNS4', 'S100A9']

    large_2 = ['AKR1C2', 'ABO', 'AQP3', 'ANXA8L1', 'ALDH3B2', 'AKR1C3', 'AKR1C1',
       'ADGRF1', 'AATBC', 'AGR2', 'ADGRF4', 'AC124789.1', 'AC003099.2',
       'ADGRV1', 'ADORA2B', 'ANK3', 'ARL4D', 'AREG', 'AC007283.5',
       'ABLIM3', 'ALDH1L1', 'CA12', 'COL17A1', 'GPX2', 'BICDL2', 'ABCA1',
       'CASP1', 'KRT14', 'KRT13', 'ACER2', 'AC011288.2', 'ATP2C2',
       'ALOX5AP', 'ANO1', 'CSTA', 'KRT5', 'AMDHD1', 'ANXA3', 'CA9',
       'CDH3', 'FAM83A', 'CKMT1A', 'C1orf116', 'CELSR2', 'GJB6', 'CKMT1B',
       'ADCY7', 'DEFB1', 'CLDN1', 'AK4', 'LY6D', 'ABI3', 'BATF2',
       'ADAMTSL4', 'AC000123.4', 'AIF1', 'ANKRD18B', 'KRT16', 'FGFR3',
       'CH17-360D5.2', 'CA2', 'AC005537.2', 'C15orf48', 'AC007326.11',
       'ACOT11', 'ALDH1A3', 'ARHGEF35', 'CCL20', 'GJB2', 'ANKRD50',
       'ADSS1', 'ABCC3', 'ARTN', 'AMN', 'ALDH5A1', 'CXCL17', 'ABAT',
       'BNIPL', 'KRT15', 'DSP', 'ANKRD22', 'AC074286.1', 'CLEC2B',
       'CXCL1', 'AFAP1L2', 'CFB', 'ADA', 'GRHL3', 'CAV1', 'AC022154.7',
       'AC004237.1', 'CXCL8', 'CNFN', 'ARNTL2', 'CXCL10', 'CELSR1',
       'IL20RB', 'ADGRG6', 'ALDH1A2', 'LCN2', 'ADAM28', 'GJB5', 'ANGPTL4',
       'ARHGEF37', 'GJB3', 'CFH', 'IVL', 'FAM110C', 'DSC2', 'CCL5',
       'C1orf210', 'C1QB', 'ARHGAP9', 'AC015849.19', 'AP1S3', 'C1orf74',
       'CARD6']
    lumP = ['S100A2', 'PSCA', 'H19', 'LY6D', 'UPK2', 'AQP3', 'ABCC3', 'KRT13', 'IGLC2', 'GPX2', 'CDC42EP5', 'VSIG2', 'DHRS2', 'SPINK1', 'TMPRSS4']

    mixed = ['IGHG1', 'IGLC2', 'CDKN2A']

    lumInf = ['IGKV3-20', 'IGLC2', 'ACTG2', 'IGHJ4', 'IGLC1', 'S100A9', 'H19', 'MUC20', 'IGHG1', 'C1QB', 'POSTN', 'GPX2', 'IGHA1', 'PSCA', 'UPK2', 'AEBP1', 'IGLC3', 'NNMT', 'KRT23', 'CTSK']

    custom_traces.append({"genes":large_2, "title": "Highest large Ba/Sq"})
    custom_traces.append({"genes":small, "title": "Highest small Ba/Sq"})

    custom_traces.append({"genes":lumP, "title": "Highest lump"})
    custom_traces.append({"genes":mixed, "title": "Highest mixed"})
    custom_traces.append({"genes":lumInf, "title": "Highest lumInf"})

    return custom_traces

def addLowSig(custom_traces):

    small = ['GSTM2', 'FABP4', 'HSD17B1', 'SCIN', 'NEB', 'AC132217.4', 'S100A3', 'RP11-20D14.6', 'SYNM', 'MYLPF', 'SOX2-OT', 'HES6', 'ALDH1A1', 'CRMP1', 'STAG3', 'CES1', 'LGALS4', 'CCL20', 'FCRLB', 'MSX1', 'SCUBE2', 'TSPAN7', 'IGF2', 'RP11-87N24.3', 'CD79B', 'CTC-425F1.4', 'ITM2A', 'AIFM3', 'NRN1', 'RP5-940J5.9', 'TGM1', 'AKR1C1', 'FABP3', 'RASD1', 'IL20RB', 'SPINK1', 'MMP9', 'SELL', 'GLDC', 'TESC', 'LINC01088', 'CD36']

    large = ['AC005301.9', 'S100A3', 'RP11-20D14.6', 'SYNM', 'MYLPF', 'SOX2-OT', 'HES6', 'ALDH1A1', 'STAG3', 'HSPB6', 'CES1', 'LGALS4', 'CCL20', 'FCRLB', 'MSX1', 'TSPAN7', 'EFHD1', 'RP11-87N24.3', 'CD79B', 'CTC-425F1.4', 'ITM2A', 'AIFM3', 'RP5-940J5.9', 'FABP3', 'RASD1', 'MMP9', 'APLP1', 'SELL', 'ID4', 'SERPINI1', 'TESC', 'LINC01088', 'CD36']

    large_2 = ['AEBP1', 'AK4', 'ADAM19', 'ALOX5AP', 'ADA', 'ANXA3', 'ADCY7',
       'AIF1', 'AREG', 'ADAMTSL4', 'ADAMTS2', 'KRT14', 'C1QB', 'ACTG2',
       'C1QC', 'CAV1', 'KRT5', 'ARNTL2', 'A1BG', 'CDH3', 'ANKH',
       'ADORA2B', 'ARL4D', 'BCL2A1', 'CCDC80', 'ANXA8L1', 'AFAP1L2',
       'CD14', 'CXCL10', 'CHI3L1', 'ALDH1A1', 'CCL4', 'CDA', 'KRT16',
       'ABI3', 'BIN1', 'CHST11', 'CXCL1', 'CASP1', 'CCL20', 'AGPAT4',
       'CTHRC1', 'AMOTL1', 'CHST15', 'CCL3', 'CA9', 'CD163', 'DSC2', 'F3',
       'C15orf48', 'COL17A1', 'BATF2', 'DSP', 'ARHGAP9', 'EFEMP1', 'AXL',
       'COL5A1', 'CCL5', 'ANPEP', 'ALDH2', 'CCDC71L', 'CXCL8', 'C12orf75',
       'COL5A2', 'CTSV', 'DEFB1', 'CD53', 'GJB2', 'COL16A1', 'IL20RB',
       'CES1', 'AC018816.3', 'FAM83A', 'COPZ2', 'CLEC2B', 'CA2',
       'AGAP2-AS1', 'C1orf162', 'CFB', 'ALDH1A3', 'ADAP2', 'AOAH',
       'ADCY9', 'AMDHD1', 'BDKRB2', 'ABCA1', 'CALHM6', 'FCGR3A',
       'CFAP251', 'ANO1', 'CD7', 'APOC2', 'AC007283.5', 'ANGPTL2', 'DPYD',
       'FAP', 'AQP3', 'ARHGEF37', 'IDO1', 'ADAMTS4', 'CD86', 'BIRC3',
       'CRYAB', 'CD300A', 'CFI', 'ARHGAP22', 'CELSR2', 'CELSR1',
       'AFAP1L1', 'CSF1R', 'CDCA7L', 'ATP10D', 'COL5A3', 'FCGR2A', 'ASPM',
       'CD3E', 'C1QTNF1', 'CCN2', 'C1orf74', 'ECM1', 'CD2', 'LAMA3',
       'CD52', 'GNLY', 'HTRA3', 'CYBB', 'FAM20A', 'CLCF1', 'CSF1',
       'LAMC2', 'CEACAM19', 'CCL2', 'CGAS', 'IGHG1', 'DSE', 'EPSTI1']


    lumP =  ['MT1M', 'HSD17B1', 'CD36', 'PTHLH', 'LMOD1', 'KRT14', 'S100A3', 'RP11-20D14.6', 'SYNM', 'CHI3L1', 'SOX2-OT', 'ALDH1A1', 'SUGCT', 'IL1R2', 'IGHG4', 'STAG3', 'OLR1', 'CES1', 'SPP1', 'CCL20', 'TSPAN7', 'EFHD1', 'GZMB', 'ANPEP', 'UCHL1', 'COMP', 'CD79B', 'ITM2A', 'AREG', 'TGM1', 'FABP3', 'EFEMP1', 'RASD1', 'IL20RB', 'MMP9', 'APLP1', 'SELL', 'IDO1', 'IGHM']

    mixed = ['MT1M', 'MMP1', 'COL17A1', 'PTHLH', 'LMOD1', 'KRT14', 'S100A3', 'SYNM', 'SOX2-OT', 'SUGCT', 'IL1R2', 'MMP7', 'CRMP1', 'OLR1', 'HSPB6', 'CES1', 'CCL20', 'CNN1', 'RP11-54H7.4', 'MYH11', 'GABRP', 'RP11-87N24.3', 'RRAD', 'CACNA2D4', 'CD79B', 'CTC-425F1.4', 'ITM2A', 'RP5-940J5.9', 'TGM1', 'FABP3', 'IL20RB', 'MMP9', 'LINC01088', 'CD36']

    lumInf = ['AC005301.9', 'DPY19L2', 'LETM2', 'CD36', 'PTHLH', 'KRT14', 'S100A3', 'RP11-20D14.6', 'RP3-467D16.3', 'SOX2-OT', 'HES6', 'ALDH1A1', 'CRMP1', 'STAG3', 'CES1', 'CCL20', 'SNORC', 'MSX1', 'RP11-54H7.4', 'RP11-87N24.3', 'CDK5R1', 'RRAD', 'CD79B', 'CTC-425F1.4', 'ITM2A', 'GLDC', 'RP5-940J5.9', 'TGM1', 'FABP3', 'RASD1', 'IL20RB', 'APLP1', 'ID4', 'IDO1', 'LINC01088', 'SCGB3A2']

    # There are a lot of genes that have low expression and are shared across. So, we're interested in the genes that are particulary low for a subgroup
    all_genes = set(small) | set(large) | set(lumP) | set(mixed) | set(lumInf) 

    custom_traces.append({"genes": set(large_2), "title": "Lowest arge Ba/Sq"})
    custom_traces.append({"genes": all_genes & set(small), "title": "Lowest small Ba/Sq"})

    custom_traces.append({"genes": all_genes & set(lumP), "title": "Lowest lump"})
    custom_traces.append({"genes": all_genes & set(mixed), "title": "Lowest mixed"})
    custom_traces.append({"genes": all_genes & set(lumInf), "title": "Lowest lumInf"})

    return custom_traces


if 0:
    #### Outside of the functions ####

    ## TCGA
    luminal_markers = ["KRT20", "PPARG", "FOXA1", "GATA3", "SNX31", "UPK1A", "UPK2", "FGFR3"]
    basal_markers = ["CD44", "KRT6A", "KRT5", "KRT14", "COL17A1"]
    squamos_markers = ["DSC3", "GSDMC", "TCGM1", "PI3", "TP63"]
    immune_markers = ["CD274", "PDCD1LG2", "IDO1", "CXCL11", "L1CAM", "SAA1"]
    neural_diff = ["MSI1", "PLEKHG4B", "GNG4", "PEG10", "RND2", "APLP1", "SOX2", "TUBB2B"]

    # TCGA markers - main paper
    emt_claudin = ["ZEB1", "ZEB2", "SNAI1", "TWIST1", "CDH2", "CLDN3", "CLDN4", "CLDN7"]
    ecm_muscle = ["PGM5", "DES", "C7", "SFRP4", "COMP", "SGCD"]

    tcga_markers = luminal_markers + basal_markers + squamos_markers + immune_markers + neural_diff + emt_claudin + ecm_muscle

    ## Urothelium
    tf_diff = ["P63", "FOXA1", "PPARG", "RARG", "IRF1", "ELF3", "GRHL3", "KLF5", "GATA4", "GATA6", "GATA3"]
    krt = ["KRT13", "KRT14", "KRT15", "KRT20"]
    upk = ["UPK1B", "UPK1A", "UPK3A", "UPK2"]
    cld = ["CLDN3", "CLDN4", "CLDN5"]

    egfr_fam = ["EGFR", "ERBB2", "ERBB3", "ERBB4", "EGF", "AREG", "HBEGF", "TGFA", "BTC", "EREG"]
    fgfr_fam = ["FGFR1", "FGFR2", "FGFR3", "FGF1", "FGF2"]
    map_kpathway = ["RAS", "RAF", "MEK1", "MEK2", "MEK3", "MEK4", "ERK"]
    pi3_kpathway = ["PIK3C3", "PIK3R2", "PIK3C2B", "AKT1", "AKT2"]
    others = ["MKI67", "MCM2", "UPK3A", "ZO1", "TJP1", "ZO2", "TJP2", "ZO3", "TJP3"]
    hox_ur = ["HOXB2", "HOXB3", "HOXB5", "HOXB6", "HOXB8"]
    hox_bla = ["HOXA9", "HOXA10", "HOXA11", "HOXA13"]

    diff_markers = tf_diff + cld + krt + upk

    uro_markers = diff_markers + egfr_fam + fgfr_fam + map_kpathway + pi3_kpathway + others + hox_ur + hox_bla

    #### Lund
    lund_qtc1 = ["FLI1", "FOXP3", "ILKZF1", "IRF4", "IRF8", "RUNX3", "SCML4", "SPI1", "STAT4", "TBX21", "TFEC"]
    lund_qtc2 = ["AEBP1", "BNC2", "GLI2", "GLIS1", "HIC1", "MSC", "PPRX1", "PPRX2", "TGFB1I1", "TWIST1"]
    lund_qtc3 = ["EBF1", "HEYL", "LEF1", "MEF2C", "TCF4", "ZEB1", "ZEB2"]
    lund_qtc8 = ["GATA5", "HAND1", "HAND2", "KLF16"]
    lund_qtc17 = ["ARID5A", "BATF3", "VENTX"]
    lund_ba_mes = lund_qtc1 + lund_qtc2 + lund_qtc3 + lund_qtc8 + lund_qtc17

    lund_ba_sq = ["BRIP1", "E2F7", "FOXM1", "ZNF367", "IRF1", "SP110", "STAT1"]
    lund_mes = ["TP53", "RB1", "FGFR3", "ANKHD1", "VIM", "ZEB2"]
    ba_sq_inf = ["CDH3", "EGFR"]

    lund_sc_ne = ["CHGA", "SYP", "ENO2", "EPCAM"]  # Highly expressed

    lund_markers = lund_ba_mes + lund_ba_sq + lund_mes + ba_sq_inf + lund_sc_ne

    b_cells = [
        "BCL2",
        "BCL6",
        "CD19",
        "CD1D",
        "CD22",
        "CD24",
        "CD27",
        "CD274",
        "CD34",
        "CD38",
        "CD40",
        "CD44",
        "CD5",
        "CD53",
        "CD69",
        "CD72",
        "CD79A",
        "CD79B",
        "CD80",
        "CD86",
        "CD93",
        "CR2",
        "CXCR4",
        "CXCR5",
        "FAS",
        "FCER2",
        "FCRL4" "HAVCR1",
        "IL10",
        "IL2RA",
        "IL7R",
        "IRF4",
        "ITGAX",
        "LILRB1",
        "MME",
        "MS4A1",
        "NT5E",
        "PDCD1LG2",
        "PRDM1",
        "PTPRC",
        "SDC1",
        "SPN",
        "TFRC",
        "TLR9",
        "TNFRSF13B",
        "TNFRSF13C",
        "TNFRSF17",
        "XBP1",
    ]
    t_cells = [
        "CD4",
        "CD8",
        "CCR4",
        "CCR5",
        "CCR6",
        "CCR7",
        "CCR10",
        "CD127",
        "CD27",
        "CD28",
        "CD38",
        "CD58",
        "CD69",
        "CTLA4",
        "CXCR3",
        "FAS",
        "IL2RA",
        "IL2RB",
        "ITGAE",
        "ITGAL",
        "KLRB1",
        "NCAM1",
        "PECAM1",
        "PTGDR2",
        "SELL",
        "IFNG",
        "IL10",
        "IL13",
        "IL17A",
        "IL2",
        "IL21",
        "IL22",
        "IL25",
        "IL26",
        "IL4",
        "IL5",
        "IL9",
        "TGFB1",
        "TNF",
        "AHR",
        "EOMES",
        "FOXO4",
        "FOXP1",
        "FOXP3",
        "GATA3",
        "IRF4",
        "LEF1",
        "PRDM1",
        "RORC",
        "STAT4",
        "TBX21",
        "TCF7",
        "GZMA",
    ]

    nk_cells = [
        "B3GAT1",
        "CCR7",
        "CD16",
        "CD2",
        "CD226",
        "CD244",
        "CD27",
        "CD300A",
        "CD34",
        "CD58",
        "CD59",
        "CD69",
        "CSF2",
        "CX3CR1",
        "CXCR1",
        "CXCR3",
        "CXCR4",
        "EOMES",
        "GZMB",
        "ICAM1",
        "IFNG",
        "IL1R1",
        "IL22",
        "IL2RB",
        "IL7R",
        "ITGA1",
        "Itga2",
        "ITGAL",
        "ITGAM",
        "ITGB2",
        "KIR2DL1",
        "KIR2DL2",
        "KIT",
        "Klrb1c",
        "KLRC1",
        "KLRC2",
        "KLRD1",
        "KLRF1",
        "KLRG1",
        "KLRK1",
        "LILRB1",
        "Klra4",
        "Klra8",
        "NCAM1",
        "NCR1",
        "NCR2",
        "NCR3",
        "PRF1",
        "SELL",
        "SIGLEC7",
        "SLAMF6",
        "SPN",
        "TBX21",
        "TNF",
    ]

    macrophages_cells = [
        "ADGRE1",
        "CCR2",
        "CD14",
        "CD68",
        "CSF1R",
        "Ly6c1",
        "MARCO",
        "MRC1",
        "NOS2",
        "PPARG",
        "SIGLEC1",
        "TLR2",
        "ARG1",
        "CD163",
        "CD200R1",
        "CD80",
        "CD86",
        "CLEC10A",
        "CLEC7A",
        "CSF2",
        "CX3CR1",
        "FCGR1A",
        "ITGAM",
        "MERTK",
        "PDCD1LG2",
        "Retnla",
        "TNF",
        "CCL22",
        "CD36",
        "CD40",
        "IL10",
        "IL1B",
        "IL6",
        "LGALS3",
        "TLR4",
        "CCL2",
        "CCR5",
        "CD209",
        "CD63",
        "CD86",
        "CSF1",
        "CXCL2",
        "FCGR3A",
        "IFNG",
        "IL4",
        "IRF4",
        "ITGAX",
        "MSR1",
        "PDGFB",
        "PTPRC",
        "STAT6",
        "TIMD4",
        "Chil3",
        "CLEC6A",
        "IL1R1",
        "ITGB2",
        "PDCD1LG2",
        "TLR7",
    ]

    monocyte_cells = [
        "CD14",
        "CD16",
        "CSF1R",
        "CX3CR1",
        "ITGAM",
        "ITGAX",
        "LY6C1",
        "CCR2",
        "CXCR4",
        "FCGR1A",
        "SELL",
        "SPN",
        "ADGRE1",
        "CCR7",
        "TNF",
        "CD86",
        "IL10",
        "IL1B",
        "MERTK",
        "TREML4",
        "CD209",
        "NR4A1",
        "Ly6a",
        "PTPRC",
        "IL3RA",
        "CD27",
        "CCR5",
        "CD32",
        "CD1A",
        "MRC1",
        "ITGB3",
        "CD9",
        "CXCR6",
        "CCR1",
        "FLT3",
        "KLF2",
        "CLEC12A",
        "CCR6",
        "CCR8",
        "CD68",
        "CLEC7A",
        "KIT",
        "MAF",
        "MAFB",
        "SPI1",
        "CD1C",
        "PPARG",
        "CEBPB",
        "ITGAE",
        "TEK",
    ]

    ig_fam = [
        "IGHG1",
        "IGHG3",
        "IGHJ2",
        "IGHJ3",
        "IGHJ4",
        "IGHM",
        "IGHV1-69D",
        "IGHV2-70",
        "IGHV3-11",
        "IGHV3-15",
        "IGHV4-39",
        "IGKC",
        "IGKJ1",
        "IGKJ4",
        "IGKV1-5",
        "IGKV1-6",
        "IGKV1-9",
        "IGKV1D-39",
        "IGKV2-28",
        "IGKV4-1",
        "IGLC1",
        "IGLC2",
        "IGLC3",
        "IGLJ2",
        "IGLJ3",
        "IGLV1-40",
        "IGLV1-44",
        "IGLV3-19",
    ]

    ifng_genes = [
        "INFG",
        "B2M",
        "BATF2",
        "BTN3A3",
        "CD74",
        "CIITA",
        "CXCL10",
        "CXCL9",
        "EPSTI1",
        "GBP1",
        "GBP4",
        "GBP5",
        "HAPLN3",
        "HLA-DPA1",
        "IDO1",
        "IFI30",
        "IFI35",
        "IFIT3",
        "IFITM1",
        "IL32",
        "IRF1",
        "NLRC5",
        "PARP9",
        "PSMB8-AS1",
        "PSMB9",
        "SAMHD1",
        "SECTM1",
        "STAT1",
        "TAP1",
        "TNFSF13B",
        "TYMP",
        "UBE2L6",
        "WARS1",
        "ZBP1",
    ]

    immune_markers = b_cells + t_cells + nk_cells + macrophages_cells + monocyte_cells


    # Basal specific
    serp_genes = ["SERPINB13", "SERPINB3", "SERPINB4"]
    spr_genes = ["SPRR1A", "SPRR1B", "SPRR2A", "SPRR2D", "SPRR3"]
    cell_diff = ["S100A2", "S100A7", "S100A8", "S100A9", "FGFBP1", "HSPB1", "KRT14", "KRT17", "KRT5", "KRT6C"]

    basal_specific = serp_genes + spr_genes + cell_diff

    data_base = 'path_to_knowles_genes'
    up_reg_scc = pd.read_csv(f"{data_base}/metadata/scc_up_regulated.csv", index_col="gene")
    down_reg_scc = pd.read_csv(f"{data_base}/metadata/scc_down_regulated.csv", index_col="gene")

    data = {
        "Luminal Markers": luminal_markers,
        "Basal Markers": basal_markers,
        "Squamous Markers": squamos_markers,
        "Immune Markers": immune_markers,
        "Neural Differentiation": neural_diff,
        "EMT and Claudin": emt_claudin,
        "ECM and Muscle": ecm_muscle,
        # urothelium
        "TF Differentiation": tf_diff,
        "KRT": krt,
        "UPK": upk,
        "CLD": cld,
        "EGFR Family": egfr_fam,
        "FGFR Family": fgfr_fam,
        "MAP Kinase Pathway": map_kpathway,
        "PI3K Pathway": pi3_kpathway,
        "Other Markers": others,
        "HOX UR": hox_ur,
        "HOX BLA": hox_bla,
        # lund
        "Lund QTc1": lund_qtc1,
        "Lund QTc2": lund_qtc2,
        "Lund QTc3": lund_qtc3,
        "Lund QTc8": lund_qtc8,
        "Lund QTc17": lund_qtc17,
        "Lund BA MES": lund_ba_mes,
        "Lund BA SQ": lund_ba_sq,
        "Lund MES": lund_mes,
        # Basal specific
        "serp": serp_genes,
        "spr": spr_genes,
        "cell_diff": cell_diff,
        "basal_specific": basal_specific,
        # immune
        "BA SQ INF": ba_sq_inf,
        "Lund SC NE": lund_sc_ne,
        "B Cells": b_cells,
        "T Cells": t_cells,
        "NK Cells": nk_cells,
        "Macrophages": macrophages_cells,
        "Monocytes": monocyte_cells,
        "ig_fam": ig_fam,
        "sb_ifng": ifng_genes,
    }

    # Convert the dictionary into a DataFrame with NaN filling for empty spaces
    df = pd.DataFrame.from_dict(data, orient="index").transpose()
    df = pd.concat([df, up_reg_scc.reset_index(names="Up Reg SCC")["Up Reg SCC"], down_reg_scc.reset_index(names="Down Reg SCC")["Down Reg SCC"]], axis=1)


    # df.to_csv(f"{figures_path}/known_markers.tsv", sep="\t", index=False)