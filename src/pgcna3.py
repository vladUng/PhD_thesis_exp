#!/usr/bin/env python
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Authors: Matthew Care, Vlad Ungureanu (3.0.0 onwards)

A memory efficient implementation of PGCNA, that requires very little RAM
to process very large expression data-sets.

This version is for processing multiple data-sets, via calculating a
median correlation matrix.

#  PGCNA2 changes  #
This is a modification of the original pgcna-multi.py that uses the
improved Leidenalg community detection method (https://arxiv.org/abs/1810.08473)
over the Louvain community detection method used originally.

"""
__author__ = "Vlad Ungureanu"
__version__ = "3.0.0"

import gzip
import os
import random
import re
import shutil
import string
import sys
from collections import defaultdict
from datetime import datetime
from math import exp

import h5py
import igraph as ig
import leidenalg as la
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.stats as stats

from NetworkAnalysis.utilities import markers as mk


class PGCNAArgs:
    # Required params
    workFolder: str  # Work Folder [REQUIRED]
    dataF: string  # Expression data folder path [REQUIRED]
    statsF: string

    # Optional params
    metaInf: str = "#FileInfo.txt"  # File containing information about data files (must be in same folder as --dataF)

    fileSep: str = "\t"  # Separator used in expression file(s)
    retainF: float = 0.8  # Retain gene fraction -- keeping most variant genes

    geneFrac: float = 1 / 3.0  # Fraction of files a gene needs to be present in to be included in median corr matrix

    edgePG: int = 3  # Edges to keep per gene -- Highly recommend leaving as default
    edgePTF: int = 50  # Edges to keep per TF
    roundL: int = 3  # Decimal places to round to before edge reduction

    #  Not often changed
    outF: str = "PGCNA"  # Root output folder
    corrMatF: str = "CORR_MATRIX"  # Correlation matrix folder
    corrMatFS: str = "CORR_MATRIX_SG"  # Folder for single gene correlation files
    gephiF: str = "GEPHI"  # Folder to store files for Gephi

    # Flags
    noLeidenalg: bool = False  # Don't try to run Leidenalg clustering, but complete everything else
    usePearson: bool = False  # Use Pearson Correlation instead of Spearman
    keepBigFA: bool = False  # Keep ALL big HDF5 files after finishing
    keepBigF: bool = False  # Keep median correlations HDF5 files after finishing
    ignoreDuplicates: bool = False  # Ignore correlation duplicates when cutting top --edgePG genes. Flag, faster if set

    # Single Corr
    singleCorr: bool = False  # Output individual gene correlation files -- Warning this generates 1 file per gene in final correlation matrix.
    singleCorrL: bool = False  # Output individual gene correlation files -- limited to those in --singleCorrListF
    singleCorrListF: str = (
        "corrGenes.txt"  # If --singleCorrL is set then create single correlation files for all those in this file (must be within --workFolder
    )

    # Corr related
    corrChunk: float = 5000  # Size of chunk (rows) to split correlation problem over [5000] -- Higher will speed up correlation calculation at cost of RAM

    #  Leidenalg (community detection) specific
    laRunNum: int = 100  # Number of times to run Leidenalg
    laBestPerc: int = 10  # Copy top [10]  %% of clusterings into lBaseFold/BEST folder
    lBaseFold: str = "LEIDENALG"  # Leidenalg base folder
    lClustTxtFold: str = "ClustersTxt"  # Leidenalg Clusters text folders
    lClustListFold: str = "ClustersLists"  # Leidenalg Clusters module list folder
    randSeed: int = 42  # Set random seed so that each run with the same settings is the same

    OUT_FOLDER: str
    expName: str

    # mutations parameters
    modifier = "standard"
    mutMetric = "count_norm"

    # Resolution paramter for constant pots
    resolution_parameter = 0.5

    # random TF genes for control
    ctrl_tfPath: str = ""

    def __init__(self, workFolder: str, dataF: str, outF: str):
        if workFolder == None or dataF == None:
            print("\n\nNeed to specifiy REQUIRED variables see help (-h)")
            sys.exit()

        self.workFolder = workFolder
        self.dataF = dataF

        self.expName = "exp-test"

        random.seed(self.randSeed)

        # setting the results folder
        self.outF = self.outF + "/" + outF
        self.OUT_FOLDER = os.path.join(self.workFolder, self.outF, "EPG" + str(self.edgePG))
        if not os.path.exists(self.OUT_FOLDER):
            os.makedirs(self.OUT_FOLDER)

        # setting the stats folder
        self.statsF = os.path.join(self.workFolder, "Stats")
        if not os.path.exists(self.statsF):
            os.makedirs(self.statsF)

        # set the meta-info file
        self.metaInf = os.path.join(self.dataF, self.metaInf)

        self.fileSep = self.fileSep.encode().decode("unicode_escape")  # string-escape in py2
        self.corrChunk = int(round(self.corrChunk, 0))
        self.laRunNum = int(round(self.laRunNum, 0))

    def setEdges(self, newEdges):
        self.edgePG = newEdges

        if self.OUT_FOLDER:
            self.OUT_FOLDER = os.path.join(self.workFolder, self.outF, "EPG" + str(self.edgePG))

        if not os.path.exists(self.OUT_FOLDER):
            os.makedirs(self.OUT_FOLDER)


class multicaster(object):
    def __init__(self, filelist):
        self.filelist = filelist

    def write(self, str):
        for f in self.filelist:
            f.write(str)


def concatenateAsString(joinWith, *args):
    temp = [str(x) for x in args]
    return joinWith.join(temp)


def overviewPrint(args: PGCNAArgs):
    print(
        "##----------------------------------------",
        str(datetime.now()),
        "----------------------------------------##",
        sep="",
    )
    #  print out settings
    settingInf = concatenateAsString(
        "\n",
        "##----------------------------------------Arguments----------------------------------------##",
        "#  Required",
        "WorkFolder [-w,--workFolder] = " + args.workFolder,
        "Expresion data file path [-d,--dataF] = " + args.dataF,
        "\n#  Optional",
        "Meta info file describing expression files [-m, --metaInf] = " + args.metaInf,
        "Separator used in expression file [-s, --fileSep] = " + args.fileSep,
        "Retain gene fraction [-f, --retainF] = " + str(args.retainF),
        "Fraction of expression files gene required in to be retained [-g, --geneFrac] = " + str(args.geneFrac),
        "Edges to retain per gene [-e, --edgePG] = " + str(args.edgePG),
        "Decimal places to round to before edge reduction [-r, --roundL] = " + str(args.roundL),
        "\n#  Main Folders" "Root output folder [--outF] = " + args.outF,
        "Correlation matrix folder [--corrMatF] = " + args.corrMatF,
        "Single Gene Correlation files folder [--corrMatFS] = " + args.corrMatFS,
        "Gephi files folder [--gephiF] = " + args.gephiF,
        "\n#  Flags",
        "Don't run Fast Unfolding clustering [--noLeidenalg] = " + str(args.noLeidenalg),
        "Use Pearson Correlation [--usePearson] = " + str(args.usePearson),
        "Keep all big HDF5 files after run [--keepBigFA] = " + str(args.keepBigFA),
        "Keep median correlations HDF5 files after run [--keepBigF] = " + str(args.keepBigF),
        "Ignore correlation duplicates when cutting top --edgePG genes [--ignoreDuplicates] = " + str(args.ignoreDuplicates),
        "Output individual gene correlation files [--singleCorr] = " + str(args.singleCorr),
        "Output individual gene correlation files for select list (--singleCorrListF) [--singleCorrL] = " + str(args.singleCorrL),
        "\n#  Single gene correlation options",
        "List of genes to process if --singleCorrL [--singleCorrListF]:\t" + str(args.singleCorrListF),
        "\n#  Correlation Options",
        "Chunk size (rows) to split correlation over [--corrChunk]:\t" + str(args.corrChunk),
        "\n#  Leidenalg Specific",
        "Random seed:\t" + str(args.randSeed),
        "Run number [-n, --laNumber]:\t" + str(args.laRunNum),
        "Copy top % of clusterings into *_BEST folder [-b, --laBestPerc]:\t" + str(args.laBestPerc),
        "Base folder [--lBaseFold] = " + str(args.lBaseFold),
        "Clusters text folder [--lClustTxtFold] = " + str(args.lClustTxtFold),
        "Clusters List folder [--lClustListFold] = " + str(args.lClustListFold),
    )

    settingsF = open(
        os.path.join(args.OUT_FOLDER, str(datetime.now()).replace(" ", "-").replace(":", ".")[:-7] + "_CorrelationSettingsInfo.txt"),
        "w",
    )
    print(settingInf, file=settingsF)
    settingsF.close()


###########################################################################################


##----------------------------------------Methods----------------------------------------##


def makeSafeFilename(inputFilename):
    #  Declare safe characters here
    safechars = string.ascii_letters + string.digits + "~-_.@#"  # string.letters in py2
    try:
        return "".join(list(filter(lambda c: c in safechars, inputFilename)))
    except:
        return ""
    pass


def make_key_naturalSort():
    """
    A factory function: creates a key function to use in sort.
    Sort data naturally
    """

    def nSort(s):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]

        return alphanum_key(s)

    return nSort


def listToPercentiles(x, inverse=False):
    data = stats.rankdata(x, "average") / float(len(x))

    if inverse:
        return (1 - data) * 100
    else:
        return data * 100


def dataSpread(x):
    """
    Returns the min, Q1 (25%), median (Q2), Q3 (75%), max, IQR, Quartile Coefficient of Dispersion and IQR/Median (CV like)
    """

    q1 = float(np.percentile(x, 25, method="lower"))
    q2 = np.percentile(x, 50)
    q3 = float(np.percentile(x, 75, method="higher"))

    if (q2 == 0) or ((q3 + q1) == 0):
        return min(x), q1, q2, q3, max(x), abs(q3 - q1), 0, 0
    else:
        return min(x), q1, q2, q3, max(x), abs(q3 - q1), abs((q3 - q1) / (q3 + q1)), abs((q3 - q1) / q2)


def mad(arr):
    """Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variabililty of the sample.
    """
    arr = np.array(arr)
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def loadCorrGeneList(workF, listFN, defaultG="IRF4"):
    print("\nLoading list of genes for single correlation file output")
    fileP = os.path.join(workF, listFN)

    if not os.path.exists(fileP):
        print("\t\t", fileP, "does not exist, will create file and add default gene (", defaultG, ")")
        outF = open(fileP, "w")
        print(defaultG, file=outF)
        outF.close()

        corrG = {defaultG: 1}

    else:
        corrG = {}
        for line in open(fileP):
            line = line.rstrip()

            corrG[line] = 1

    print("\tTotal genes:", len(corrG))

    return corrG


def generateGeneMeta(outF, fileN, geneMetaInfo, genesNP, npA):
    """
    Generate meta information for gene
    """

    ###########################################################################################
    def expToPercentiles(expToPerc, expA):
        return [expToPerc[e] for e in expA]

    ###########################################################################################
    print("\t\t# Generate gene meta information : ", end="")

    expressionVals = npA.flatten()

    #  Convert from expression values to percentiles
    expPercentiles = listToPercentiles(expressionVals, inverse=False)

    #  Calculate mapping from expression to percentile
    expToPerc = {}
    for i, e in enumerate(expressionVals):
        expToPerc[e] = expPercentiles[i]

    outF = open(os.path.join(outF, fileN + "_metaInf.txt"), "w")
    for i, gene in enumerate(genesNP):
        #  Convert from expression values to percentiles, so can work out (mean absolute deviation) MAD of percentiles
        genePercentiles = expToPercentiles(expToPerc, npA[i])

        #  min, Q1 (25%), median (Q2), Q3 (75%), max, IQR, Quartile Coefficient of Dispersion and IQR/Median (CV like
        try:
            minE, q1E, q2E, q3E, maxE, iqrE, qcodE, iqrME = dataSpread(npA[i])
        except:
            print("Issue with :", gene, npA[i])
            sys.exit()

        medianPercentiles = np.median(genePercentiles)
        # quartile to expression, median (Q2) varWithin, varAcross -  Quartile Coefficient of Dispersion
        print(gene, q2E, qcodE, medianPercentiles, sep="\t", file=outF)

        geneMetaInfo[gene].append([q2E, qcodE, medianPercentiles])

    outF.close()
    print("done")


def mergeMetaInfo(geneMetaInfo):
    print("\n\tMerging gene meta information")
    medianMetaInfo = {}

    for gene in geneMetaInfo:
        medians, qcods, percentiles = zip(*geneMetaInfo[gene])
        #  Store Median_QCOD(varWithin), MedianPercentile, MADpercentile(VarAcross)
        medianMetaInfo[gene] = [np.median(qcods), np.median(percentiles), mad(percentiles)]

    return medianMetaInfo


def loadMeta(metaFile, dataF, splitBy="\t", headerL=1):
    print("\n\nLoad meta-file (", os.path.basename(metaFile), ")", sep="")
    if not os.path.exists(metaFile):
        print("\t\t# Meta file (", metaFile, ") does not exist!", sep="")
        sys.exit()

    header = headerL
    fileInfo = {}
    for line in open(metaFile):
        cols = line.rstrip().split(splitBy)

        if header:
            header -= 1
            continue

        if line.rstrip() == "":
            continue

        totalPath = os.path.join(dataF, cols[0])
        if not os.path.exists(totalPath):
            print("\t\t# File (", totalPath, ") does not exist!, won't add to fileInfo", sep="")
        else:
            try:
                fileInfo[totalPath] = int(cols[1])
            except:
                print("Meta file line (", line.rstrip(), ") is not formed properly, skipping")

    print("\tLoaded information on:", len(fileInfo), "files")
    return fileInfo


def getGenes(metaInf, splitBy="\t", geneFrac=1 / 3.0, retainF=0.8):
    """
    First pass over expression data-files to find the set of genes we will be
    working with after filtering each data-set by retainF and then only keeping
    genes that are present in geneFrac
    """
    natSort = make_key_naturalSort()

    geneCount = defaultdict(int)

    print("\nLoad information on genes present per expression array:")
    for fileP in sorted(metaInf, key=natSort):
        fileN = os.path.basename(fileP)

        print("\t", fileN, end="")

        headerL = metaInf[fileP]
        #############################
        # Read in data file
        seenGenes = {}
        genes = []
        expression = []

        print("\tReading expression data file:", fileN)

        for i, line in enumerate(open(fileP)):
            cols = line.rstrip().replace('"', "").replace("'", "").replace(" ", "").split(splitBy)

            if headerL:
                headerL -= 1
                continue

            genes.append(cols[0])
            if not cols[0] in seenGenes:
                seenGenes[cols[0]] = 1
            else:
                print(
                    "\n\nERROR: Duplicate gene (",
                    cols[0],
                    ") in expression data (",
                    fileN,
                    "), expects unique identifiers!\nPlease remove duplicates and re-run\n",
                    sep="",
                )
                sys.exit()
            expression.append(cols[1:])

        print("\t\tTotal genes = ", len(genes))
        #############################

        #############################
        print("\t\t# Generate Numpy matrices")
        #  Move genes to numpy array, so can be sorted along with the rest
        genesNP = np.empty((len(genes)), dtype=str)
        genesNP = np.array(genes)

        #  Create numpy array
        nrow = len(expression[0])
        ncol = len(expression)
        npA = np.empty((ncol, nrow), dtype=object)
        for i, eSet in enumerate(expression):
            npA[i] = eSet

        #  Calculate SD for each gene
        npSD = np.std(npA.astype(np.float64), axis=1)

        #  Sort by SD
        sortI = np.argsort(npSD)[::-1]  # Sort ascending
        #  Sort matrix by index
        npA = npA[sortI, :]

        #  Sort genes by index
        genesNP = genesNP[sortI]

        # Important: Here is where the number of genes are cut

        #  Only retain top fract #
        #  Work out index to cut at
        cutPos = int(round(retainF * genesNP.shape[0], 0))
        print("\t\tRetain Fraction (", retainF, ") Cut at: ", cutPos, sep="")
        print("\t\t\tPre-cut shape:", npA.shape)

        #  Retain most variant
        npA = npA[:cutPos,].astype(np.float64)
        genesNP = genesNP[:cutPos]
        print("\t\t\tPost-cut shape:", npA.shape)

        #  Keep track of seen genes
        for gene in genesNP:
            geneCount[gene] += 1

    print("\n\tTotal unique genes/identifiers:", len(geneCount))

    requiredFileNum = int(round(geneFrac * len(metaInf), 0))
    print("\tRetaining genes present in ", round(geneFrac, 2), " (>=", requiredFileNum, ") of files : ", end="", sep="")

    genesToKeep = {}
    for g in geneCount:
        if geneCount[g] >= requiredFileNum:
            genesToKeep[g] = 1

    print(len(genesToKeep))

    return genesToKeep, requiredFileNum


def loadHDF5files(outF, hdf5Paths):
    hdf5Files = {}
    for fileN in hdf5Paths:
        hdf5Files[fileN] = h5py.File(fileN, "r")

    return hdf5Files


def generateMasks(sortedKeepGenes, genesPerHDF5, hdf5paths):
    print("\tGenerating index mappings and masks to speed up combining correlations")

    masks = np.zeros((len(hdf5paths), len(sortedKeepGenes)))
    for i, g in enumerate(sortedKeepGenes):
        for j, hd5 in enumerate(hdf5paths):
            if g not in genesPerHDF5[hd5]:
                masks[j][i] = 1  # Mask missing values
    return masks


def generateCorrMatrix(
    workF,
    metaInf,
    genesToKeep,
    requiredFileNum,
    singleCorrList,
    geneFrac=1 / 3.0,
    retainF=0.8,
    corrMatF="CORR_MATRIX",
    corrMatFS="CORR_MATRIX_SG",
    usePearson=False,
    corrChunk=5000,
    splitBy="\t",
    decimalP=3,
    printEveryDiv=10,
    keepBigFA=False,
    singleCorr=False,
    singleCorrL=False,
):
    """
    Memory efficient method for generating all pairwise correlations of genes (rows) across a set of samples (columns).  Uses HDF5 files to greatly
    reduce memory useage, keeping most data residing on the hard disk.
    """
    print("\n\nGenerate correlations for expression data:")
    natSort = make_key_naturalSort()
    sortedKeepGenes = sorted(genesToKeep, key=natSort)

    #  Find mapping of keep genes
    keepGeneMapping = {}
    for i, g in enumerate(sortedKeepGenes):
        keepGeneMapping[g] = i

    # Create subfolder
    outF = os.path.join(workF, corrMatF + "_GMF" + str(requiredFileNum))
    if not os.path.exists(outF):
        os.makedirs(outF)

    if singleCorr or singleCorrL:
        outFS = os.path.join(workF, corrMatFS + "_GMF" + str(requiredFileNum))
        if not os.path.exists(outF):
            os.makedirs(outF)

    geneMetaInfo = defaultdict(list)  # Per data-set store gene meta information
    genePosPerHDF5 = defaultdict(dict)  # Mapping of location of gene in HDF5 file
    perHDF5index = defaultdict(list)  # Mapping of gene to where they should be in final matrix
    genesPerHDF5 = defaultdict(dict)  # Keep tally of which genes are in which HDF5 file
    hdf5paths = []

    print("\n\tCalculating correlations for data file:")
    for fileP in sorted(metaInf, key=natSort):
        fileN = os.path.basename(fileP)

        print("\t", fileN, end="")

        headerL = metaInf[fileP]
        #############################
        # Read in data file
        seenGenes = {}
        genes = []
        expression = []

        print("\tReading expression data file:", fileN)

        for i, line in enumerate(open(fileP)):
            cols = line.rstrip().replace('"', "").replace("'", "").replace(" ", "").split(splitBy)

            # Skip if it's the header
            if headerL:
                headerL -= 1
                continue

            #  Only retain required genes
            if cols[0] not in genesToKeep:
                continue

            genes.append(cols[0])
            if not cols[0] in seenGenes:
                seenGenes[cols[0]] = 1
            else:
                print(
                    "\n\nERROR: Duplicate gene (",
                    cols[0],
                    ") in expression data (",
                    fileN,
                    "), expects unique identifiers!\nPlease remove duplicates and re-run\n",
                    sep="",
                )
                sys.exit()
            expression.append(cols[1:])

        print("\t\tTotal genes = ", len(genes))
        #############################

        #############################
        print("\t\t# Generate Numpy matrices")
        #  Move genes to numpy array
        genesNP = np.empty((len(genes)), dtype=str)  # redundant by the next line
        genesNP = np.array(genes)

        #  Store position of gene in matrix for this file and create mapping index for HDF5 files
        outMatrixHDF5 = os.path.join(outF, fileN + "_RetainF" + str(retainF) + ".h5")
        hdf5paths.append(outMatrixHDF5)

        for i, g in enumerate(genesNP):
            perHDF5index[outMatrixHDF5].append(keepGeneMapping[g])
            genePosPerHDF5[g][outMatrixHDF5] = i
            genesPerHDF5[outMatrixHDF5][g] = 1

        #  Create numpy array
        nrow = len(expression[0])
        ncol = len(expression)
        npA = np.zeros((ncol, nrow), dtype=np.float64)
        for i, eSet in enumerate(expression):
            npA[i] = eSet

        print("\t\t\tMarix shape:", npA.shape)

        #  Output genes
        outMatrixN = os.path.join(outF, fileN + "_RetainF" + str(retainF) + "_Genes.txt")
        np.savetxt(outMatrixN, genesNP, fmt="%s", delimiter="\t")

        #  Generate gene meta information
        #  Each key is a gene which has a list of 3 values: [median, Quartilee Coefficient, medianPeercentilees]
        generateGeneMeta(outF, fileN + "_RetainF" + str(retainF), geneMetaInfo, genesNP, npA)

        #######################################
        # Calculate correlations
        print("\t\t# Calculating correlations using HDF5 file to save memory")
        rowN, colN = npA.shape

        if os.path.exists(outMatrixHDF5):
            # if False:
            print("\t\tAlready exists -- skipping")
            continue
        with h5py.File(outMatrixHDF5, "w") as f:
            h5 = f.create_dataset("corr", (rowN, rowN), dtype="f8")

        # # Load into memory
        print("\t\t\tCreating HDF5 file")
        h5 = h5py.File(outMatrixHDF5, "r+")

        #### Calculating spearman
        if not usePearson:
            # Note this isn't a perfect Spearman, as ties are not dealt with, but it is fast, and given we're only
            # Retaining the most correlated edges will have almost no effect on the output.
            npA = npA.argsort(axis=1).argsort(axis=1).astype(np.float64)

        # subtract means from the input data
        npA -= np.mean(npA, axis=1)[:, None]

        # normalize the data
        npA /= np.sqrt(np.sum(npA * npA, axis=1))[:, None]

        # Calculate correlations per chunk
        print("\t\t\tCalculating correlations for Chunk:")
        for r in range(0, rowN, corrChunk):
            print("\t\t\t", r)
            for c in range(0, rowN, corrChunk):
                r1 = r + corrChunk
                c1 = c + corrChunk
                chunk1 = npA[r:r1]
                chunk2 = npA[c:c1]
                h5["corr"][r:r1, c:c1] = np.dot(chunk1, chunk2.T)

        # # Write output out for debugging
        # finalCorr = np.copy(h5["corr"][:])
        # outMatrixN = os.path.join(outF,fileN + "_RetainF" + str(retainF) + "_Corr")
        # stringFormat = "%." + str(decimalP) + "f"
        # # stringFormat2 = "%." + str(decimalPpVal) + "f"
        # np.savetxt(outMatrixN + ".txt",finalCorr,fmt=stringFormat,delimiter="\t")

        #  Remove matrices to save memory
        del npA
        del h5

    #  Calculate median gene meta information
    medianMetaInfo = mergeMetaInfo(geneMetaInfo)

    ###########################################################################################
    ##----------------------------Calculate Median Correlations------------------------------##
    ###########################################################################################
    #  Calculate median correlation matrix
    print("\nCalculating median correlations")
    outMatrixHDF5_orig = outMatrixHDF5  # Retain incase only single data-set
    outMatrixHDF5 = os.path.join(outF, "#Median_RetainF" + str(retainF) + ".h5")
    if not os.path.exists(outMatrixHDF5):
        if len(hdf5paths) == 1:
            #  If we're only analysing a single data-set
            if not (singleCorr or singleCorrL):
                print("\t\tOnly single data-set analysed, skipping generating median correlations")
                #  No need to calculate median correlations, just return path to HDF5 file and genes matrix
                return outMatrixHDF5_orig, genesNP
            else:
                print("\t\tOnly single data-set analysed, but --singeCorr/--singleCorrL so will proceed with output")

        printEvery = int(round(len(genesToKeep) / float(printEveryDiv), 0))
        printE = printEvery
        tell = printE
        count = 0

        #  Create HDF5 median correlation matrix

        with h5py.File(outMatrixHDF5, "w") as f:
            h5 = f.create_dataset("corr", (len(genesToKeep), len(genesToKeep)), dtype="f8")
        h5 = h5py.File(outMatrixHDF5, "r+")

        # Load HDF5 correlation files
        hdf5Files = loadHDF5files(outF, hdf5paths)

        #  Output genes
        outMatrixN = os.path.join(outF, "#Median_RetainF" + str(retainF) + "_Genes.txt")
        np.savetxt(outMatrixN, sortedKeepGenes, fmt="%s", delimiter="\t")

        #  Get masks
        maskMappings = generateMasks(sortedKeepGenes, genesPerHDF5, hdf5paths)

        print("\tCalculating median correlations (report every 1/", printEveryDiv, "th of total):", sep="")
        if singleCorr or singleCorrL:
            print("\t\tOutputting single gene correlation files:")

        for genePos, gene in enumerate(sortedKeepGenes):
            # print("\t\t",gene1)
            #  Inform user of position
            count += 1
            if count == printE:
                printE = printE + tell
                if singleCorr:
                    print("\n\t\tProcessed:", count, end="\n\n")
                else:
                    print("\t\t", count)

            rowsPerHDF5 = {}
            maskPos = []
            dataSetNames = []
            #  Grab row for gene across files
            for i, hdf5 in enumerate(hdf5paths):
                try:
                    rowsPerHDF5[hdf5] = hdf5Files[hdf5]["corr"][genePosPerHDF5[gene][hdf5]]
                    maskPos.append(i)
                    dataSetNames.append(os.path.basename(hdf5)[:-3])
                except:
                    pass

            #  Second pass
            #  Convert to numpy array
            npA = np.full((len(rowsPerHDF5), len(sortedKeepGenes)), -10, dtype=np.float64)  # Missing set to -10
            for i, hdf5 in enumerate(sorted(rowsPerHDF5, key=natSort)):
                npA[i][perHDF5index[hdf5]] = rowsPerHDF5[hdf5]  # Use indexes to place in correct location

            #  Get appropriate masks
            tempMask = []
            for i in maskPos:
                tempMask.append(maskMappings[i])

            npAMasked = ma.masked_array(npA, mask=tempMask)

            #  Generate medians
            medianRowCorr = np.copy(ma.median(npAMasked, axis=0))
            h5["corr"][genePos] = medianRowCorr

            ###########################################################################################
            ##-------------------------------SINGLE GENE CORR----------------------------------------##
            ###########################################################################################
            def mergeCorrData(gene, corrInf, medianMetaInfo, medianCorr, missingVal=-10):
                finalCorrs = []
                dataSetCount = 0

                for corr in corrInf:
                    if (corr <= missingVal).all():
                        finalCorrs.append("")
                    else:
                        finalCorrs.append(str(round(corr, decimalP)))
                        dataSetCount += 1

                roundedMeta = map(str, [round(origV, decimalP) for origV in medianMetaInfo[gene]])
                scaledCorr = dataSetCount**medianCorr
                finalInfo = (
                    gene
                    + "\t"
                    + str(round(scaledCorr, decimalP))
                    + "\t"
                    + str(dataSetCount)
                    + "\t"
                    + str(round(medianCorr, decimalP))
                    + "\t"
                    + "\t".join(roundedMeta)
                    + "\t"
                    + "\t".join(finalCorrs)
                )

                return scaledCorr, finalInfo

            ###################################
            if singleCorr or singleCorrL:
                if singleCorrL:  # Only output genes in list
                    if gene not in singleCorrList:
                        continue

                subFolder = os.path.join(outFS, gene[0])
                singleCorrFName = os.path.join(subFolder, str(makeSafeFilename(gene)) + "_corr_RetainF" + str(retainF) + ".txt.gz")
                if os.path.exists(singleCorrFName):
                    continue

                print("\t\t\tSingle Corr:", str(makeSafeFilename(gene)))

                #  Make subfolder
                if not os.path.exists(subFolder):
                    os.makedirs(subFolder)

                singleCorrF = gzip.open(singleCorrFName, "wt", compresslevel=9)  # "wb" in py2
                dataSetsH = "\t".join(dataSetNames)

                if usePearson:
                    print(
                        "Gene\tNumDataSets^MedianPCC\tNumDataSets\tMedianPCC\tMedian_QCODexpression(VarWithin)\tMedianPercentile\tMADpercentile(VarAcross)",
                        dataSetsH,
                        sep="\t",
                        file=singleCorrF,
                    )
                else:
                    print(
                        "Gene\tNumDataSets^MedianRho\tNumDataSets\tMedianRho\tMedian_QCODexpression(VarWithin)\tMedianPercentile\tMADpercentile(VarAcross)",
                        dataSetsH,
                        sep="\t",
                        file=singleCorrF,
                    )

                rankedByCorr = defaultdict(list)
                for i, g in enumerate(sortedKeepGenes):
                    scaledCorr, info = mergeCorrData(g, npA.T[i], medianMetaInfo, medianRowCorr[keepGeneMapping[g]])
                    rankedByCorr[scaledCorr].append(info)

                #  Rank by scaledCorr
                for sCorr in sorted(rankedByCorr, reverse=True):
                    for info in rankedByCorr[sCorr]:
                        print(info, file=singleCorrF)
                singleCorrF.close()

        ###########################################################################################
        # # Write output out for debugging
        # finalCorr = np.copy(h5["corr"][:])
        # outMatrixN = os.path.join(outF,"#Median_RetainF" + str(retainF) + "_Corr")
        # stringFormat = "%." + str(decimalP) + "f"
        # np.savetxt(outMatrixN + ".txt",finalCorr,fmt=stringFormat,delimiter="\t")
        ###########################################################################################
        ###########################################################################################

        #  Remove all single HDF5 files unless requested to keep them
        if not keepBigFA:
            del hdf5Files
            for hdf5P in hdf5paths:
                os.remove(hdf5P)
    else:
        print("\t\tAlready exists -- skipping")

    #  Return path to HDF5 file and genes matrix - which is a sorted gene matrix
    return outMatrixHDF5, sortedKeepGenes


def integrateMutations(dataF, corrh5, genesM, keepBigF, mutMetric="count_norm", modifier_type="standard"):
    def rescaled(series: pd.Series, new_max: int):
        """Rescales a given series to a new range.

        Args:
            series (pd.Series): The series to be rescaled.
            new_max (int): The maximum value of the new range.

        Returns:
            pd.Series: The rescaled series.
        """
        new_min = -new_max
        x_norm = (series - series.min()) / (series.max() - series.min())
        x_scaled = x_norm * (new_max - new_min) + new_min

        return x_scaled

    def sigmoid_func(x, x0, offset=1):
        """
        Sigmoid function

        Args:
            x (int/float): The variable
            x0 (center): Where the sigmoid is centered
            offset (int, optional): The elongation on y-axis. Defaults to 1.

            By default the sigmoid function starts goes from 1 to 2. The offset elongates on the y axis

        Returns:
            _type_: _description_
        """
        return (1 + exp(-(x - x0))) ** -1 * offset + 1

    def loadMutations(path, genes, sortCol="count"):
        mut_df = pd.read_csv(path + "/post/mut_tcgna_pgcna_v1.csv", index_col="gene")

        filtered_df = pd.DataFrame(index=genes)
        filtered_df = pd.concat([filtered_df, mut_df.loc[mut_df.index.isin(genes)]], axis=1).fillna(0)

        # important to do it after we filtered the genes
        x = np.log2(mut_df["count"] + 1)
        max_x = x.max()

        filtered_df["beta"] = (max_x - x) / max_x
        filtered_df["norm3"] = (max_x + x) / max_x
        filtered_df["norm3"] = filtered_df["norm3"].fillna(1)

        if "sigmoid" in sortCol:
            # 12, -12 was found through experimentations
            x0, offset = -8, 1
            filtered_df["rescaled"] = rescaled(filtered_df["count"], new_max=12)
            filtered_df[sortCol] = filtered_df["rescaled"].apply(sigmoid_func, args=(x0, offset))

        return filtered_df.sort_values(by=[sortCol]).round(6)

    def applyModifier(row, modifier, modifier_type):
        new_row = row
        if modifier_type == "norm":
            new_row = new_row * (modifier + new_row)
        elif modifier_type == "norm2":
            new_row = new_row * (modifier + 1)
        elif modifier_type == "beta" or modifier_type == "norm3":
            new_row = new_row * modifier
        # Below v2 for weight modifiers
        elif modifier_type == "sigmoid_v1":
            new_row = row * modifier
        return new_row

    # do nothing
    if modifier_type == "standard":
        return corrh5

    print("\tLoad HDF5 file")

    if keepBigF:
        #  Backup original raw correlations before edge reduction
        corrh5_raw = corrh5[:-3] + "_NoMutations.h5"
        shutil.copy(corrh5, corrh5_raw)

    # Reading the corr matrix
    h5 = h5py.File(corrh5, "r+")

    # Reading the mutations DataFrame
    # mutMetric, modifier = "count_norm", "n"
    mut_df = loadMutations(dataF, genesM, sortCol=mutMetric)

    print("\t### Integrating mutations count to the correlation matrix ###")
    # TODO THis can be further improved by doing matrix multiplication
    #  1. Sort the mut DataFrame to be in sync with the numpy array
    #  2. Propagate the mut count across
    for i, row in enumerate(h5["corr"]):
        gene = genesM[i]
        modifier = mut_df.loc[gene][mutMetric]

        h5["corr"][i] = applyModifier(row, modifier, modifier_type=modifier_type)

    # TODO Don't we need closing in each situations?
    #  Kept the above code to be consistent
    if not (keepBigF):
        h5.close()

    return corrh5


def reduceEdges(
    workF,
    dataF,
    gephiF,
    corrh5,
    genesM,
    markerGenes,
    retainF=0.8,
    edgePG=3,
    edgePTF=50,
    printEveryDiv=10,
    corrMatF=None,
    keepBigFA=False,
    keepBigF=False,
    ignoreDuplicates=False,
    roundL=3,
    tfList=None,
):
    """
    Reduce edges in correlation matrix, only retaining edgePG maximum correlated genes per row
    """

    def bottomToZero(npA, n=1):
        """
        Set everything below n to zero
        """
        topI = np.argpartition(npA, -n)
        npA[topI[:-n]] = 0
        return npA

    def bottomToZeroWithDuplicates(npA, n=1):
        """
        Set everything below n to zero,
        but deal with duplicates
        """
        unique = np.unique(npA)

        uniqueGTzero = len(unique[unique > 0])
        if n > uniqueGTzero:
            #  Deal with edgePG extending into negative correlations
            n = uniqueGTzero

        topIunique = np.argpartition(unique, -n)[-n:]
        toKeep = []
        for val in unique[topIunique]:
            # this ensures that the duplictates are kept too
            toKeep.extend(np.where(npA == val)[0])

        #  Mask and reverse
        mask = np.ones(len(npA), bool)
        mask[toKeep] = 0
        npA[mask] = 0

        return npA

    ###########################################################################################
    #  Setup #
    print("\nReduces edges to (", edgePG, ") per gene:", sep="")
    fileN = os.path.basename(dataF)
    #  Create subfolder
    outF = os.path.join(workF, gephiF)
    if not os.path.exists(outF):
        os.makedirs(outF)
    nodesFP = os.path.join(outF, fileN + "_RetainF" + str(retainF) + "_EPG" + str(edgePG) + "_Nodes.tsv.gz")
    edgesFP = os.path.join(outF, fileN + "_RetainF" + str(retainF) + "_EPG" + str(edgePG) + "_Edges.tsv.gz")
    # if os.path.exists(nodesFP) and os.path.exists(edgesFP):
    #     print("\t\tEdge reduction files already exist, skipping")
    #     return edgesFP, nodesFP
    ###########################################################################################

    print("\tLoad HDF5 file")

    if keepBigFA or keepBigF:
        #  Backup original raw correlations before edge reduction
        corrh5_raw = corrh5[:-3] + "_NoEdgeReduction.h5"
        shutil.copy(corrh5, corrh5_raw)

    #  Load old HDF5
    h5 = h5py.File(corrh5, "r+")
    rowN, colN = h5["corr"].shape

    printEvery = int(round(rowN / float(printEveryDiv), 0))

    printE = printEvery
    tell = printE
    count = 0

    print("\tWorking (report every 1/", printEveryDiv, "th of total):", sep="")
    for i, row in enumerate(h5["corr"]):
        #  Inform user of position
        count += 1
        if count == printE:
            printE = printE + tell
            print("\t\t", count)

        #  Before edge reduction round correlations
        row = np.round(row, decimals=roundL)

        ### Change here for edge reduction and keep more for TF
        if ignoreDuplicates:
            h5["corr"][i] = bottomToZero(row, edgePG + 1)
        elif genesM[i] in tfList:
            h5["corr"][i] = bottomToZeroWithDuplicates(row, edgePTF + 1)
        else:
            h5["corr"][i] = bottomToZeroWithDuplicates(row, edgePG + 1)
        
        # if genesM[i] in ["FOXQ1", "CEBPA"]:
        #     test = row.copy()
        #     test2 = h5["corr"][i].copy()
        #     test2.sort()
        #     test.sort()
        #     print(genesM[i])
        #     print(test[-10:])
        #     print(test2[-10:])
        #     print("sada")

    # Write output out for debugging
    # finalCorr = np.copy(h5EPG["corr"][:])
    # outMatrixN = os.path.join(outF,fileN + "_RetainF" + str(retainF) + "_NonSym")
    # stringFormat = "%." + str(3) + "f"
    # # stringFormat2 = "%." + str(decimalPpVal) + "f"
    # np.savetxt(outMatrixN + ".txt",finalCorr,fmt=stringFormat,delimiter="\t")

    ###########################################################################################
    print("\nGenerating files for Gephi network visualization tool")

    #  First output list of genes (nodes)
    print("\tCreate node file")

    nodesFile = gzip.open(nodesFP, "wt", compresslevel=9)
    # print("Id\tLabel\tCluster", file=nodesFile)
    print("Id\tLabel\tVU\tVU_200\tVU_500\tVU_rel_200\tVU_rel_500\tLund\tTCGA\tTF", file=nodesFile)

    for gene in genesM:
        # Important: Change here to add the cluster
        row = markerGenes.loc[gene]
        print(
            gene,
            row["VU_rel_50_gc42"],
            row["VU_rel_500_gc42"],
            row["VU_rel_200_gc42"],
            row["VU_rel_100_gc42"],
            row["Lund"],
            row["TCGA"],
            row["TF"],
            file=nodesFile,
            sep="\t",
        )
    nodesFile.close()

    #  Second, output list of edges
    print("\tCreate edges file (report every 1/", printEveryDiv, "th of total):", sep="")
    edgesFile = gzip.open(edgesFP, "wt", compresslevel=9)
    print("Source\tTarget\tWeight\tType\tfromAltName\ttoAltName", file=edgesFile)

    # Finally output edges
    printE = printEvery
    tell = printE
    count = 0
    seen = defaultdict()
    printedLines = 0

    # Here we write in the edges file...
    for i, row in enumerate(h5["corr"]):
        for j, item in enumerate(row):
            if not (i == j):  # Don't output self edges...
                geneO = genesM[i]
                geneT = genesM[j]

                if not (geneO + "-@@@-" + geneT) in seen:
                    #  Output info
                    if not item == 0:
                        print(geneO, geneT, item, "undirected", geneO, geneT, sep="\t", file=edgesFile)
                        printedLines += 1

                        #  Store the fact that we've see this and it's equivalent pair
                        seen[geneO + "-@@@-" + geneT] = 1
                        seen[geneT + "-@@@-" + geneO] = 1

        count += 1
        if count == printEvery:
            printEvery = printEvery + tell
            print("\t\t", count, "Genes", ", Edges:", printedLines)

    edgesFile.close()
    print("\n\t\tTotal printed (", printedLines, ") edges", sep="")

    if not (keepBigFA or keepBigF):
        h5.close()

        # get the folder where the h5 files are
        folder_path = "/".join(corrh5.split("/")[:-1]) + "/"
        files = next(os.walk("/".join(corrh5.split("/")[:-1]) + "/"), (None, None, []))[2]
        # just remove the h5 files
        for file in files:
            if ".h5" in file:
                os.remove(folder_path + file)
            else:
                continue
        # shutil.rmtree(os.path.join(workF, corrMatF))

    return edgesFP, nodesFP


def runleidenalgF(
    wF,
    statsF,
    edgesFP,
    baseFold="LEIDENALG",
    outFoldTorig="ClustersTxt",
    outFoldLorig="ClustersLists",
    runNum=10,
    bestPerc=10,
    printEveryDiv=10,
    removeExisting=True,
    allFold="ALL",
    bestFold="BEST",
    meta={},
    resolution_parameter=0.5,
    tfList=[],
):
    """
    Convert edges into igraph (https://igraph.org/python/) format and then carry out community detection using the python Leidenalg package (https://github.com/vtraag/leidenalg).  This is an improvement
    upon the Louvain method used in pgcna.py/pgcna-multi.py that ensures that all communities are well connected within the network.
    """
    ###########################################################################################

    def convertGephiToIgraph(gephiEdgeFile, splitBy="\t", header=1, outN="igraphG.temp"):
        """
        Convert edge file into format for import to igraph
        """
        baseFold, fileName = os.path.split(gephiEdgeFile)
        outFileP = os.path.join(baseFold, outN)
        outFile = open(outFileP, "w")

        for line in gzip.open(gephiEdgeFile):
            line = line.decode()
            if header:
                header -= 1
                continue

            cols = line.split(splitBy)
            print(" ".join(cols[0:3]), file=outFile)

        outFile.close()
        return outFileP

    def outputModules(outFoldT, outFoldL, partition, runNum=1):
        # Natural sort
        natSort = make_key_naturalSort()

        names = partition.graph.vs["name"]
        members = partition.membership
        namePerMod = defaultdict(dict)

        for n, m in zip(names, members):
            namePerMod[int(m) + 1][n] = 1

        #  Output module lists
        outFileT = open(os.path.join(outFoldT, str(runNum) + ".csv"), "w")
        print("Id,Modularity Class", file=outFileT)
        for m in sorted(namePerMod):
            outFileL = open(os.path.join(outFoldL, "M" + str(m) + ".txt"), "w")
            for n in sorted(namePerMod[m], key=natSort):
                print(n, file=outFileL)
                print(n, m, sep=",", file=outFileT)
            outFileL.close()
        outFileT.close()
        return namePerMod

    def saveStats(g, meta, bestRuns, tfList, path):
        import pickle

        import pandas as pd

        # this holds a history of the experiments run
        masterFile = "stats_master.tsv"
        masterDf = pd.DataFrame()
        masterFilePath = os.path.join(path, masterFile)
        if os.path.exists(masterFilePath):
            masterDf = pd.read_csv(masterFilePath, sep="\t")

        # newEntry = pd.concat([pd.DataFrame.from_dict(bestRuns[1]["stats"]), pd.DataFrame.from_dict(meta)])

        # update the master file
        masterDf = pd.concat([masterDf, pd.DataFrame.from_dict(meta)])
        masterDf.to_csv(masterFilePath, sep="\t", index=False)

        # add data to pickle
        meta["graph"] = g
        meta["bestRuns"] = bestRuns
        meta["tf_list"] = tfList

        # save the data to pickle
        filename = meta["expName"] + ".pickle"
        objPath = os.path.join(path, filename)
        with open(objPath, "wb") as handle:
            pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###########################################################################################

    print("\nGenerating Leidenalg clusterings (n=", runNum, ")", sep="")

    #  Create output folder
    if removeExisting:
        baseL = os.path.join(wF, baseFold)
        if os.path.exists(baseL):
            shutil.rmtree(baseL)

    outFoldT = os.path.join(wF, baseFold, allFold, outFoldTorig)
    outFoldL = os.path.join(wF, baseFold, allFold, outFoldLorig)
    if not os.path.exists(outFoldT):
        os.makedirs(outFoldT)
    if not os.path.exists(outFoldL):
        os.makedirs(outFoldL)

    tempP = convertGephiToIgraph(edgesFP)
    #  Input graph
    print("\tConverting gephi edge file --> igraph edge file")
    g = ig.Graph().Read_Ncol(tempP, directed=False)
    os.remove(tempP)

    # Get weights
    graphWeights = g.es["weight"]

    infoPerClustering = []
    modInfo = []
    modInfoScores = defaultdict(list)

    printEvery = int(round(runNum / float(printEveryDiv), 0))
    printE = printEvery
    tell = printE
    count = 0

    print("\tWorking (report every 1/", printEveryDiv, "th of total):", sep="")
    for rNum in range(1, runNum + 1):
        #  Make sub folder
        outFoldLS = os.path.join(outFoldL, "Clust" + str(rNum))
        if not os.path.exists(outFoldLS):
            os.makedirs(outFoldLS)

        seed = random.randint(0, 1e9)  # Generate random seed as leidenalg doesn't appear to do this (even though it says it does)
        partition = la.find_partition(
            g, la.ModularityVertexPartition, weights=graphWeights, n_iterations=-1, seed=seed
        )  # n_iterations=-1 to converge on best local result

        # partition = la.find_partition(
        #     g, la.CPMVertexPartition, weights=graphWeights, n_iterations=-1, seed=seed,
        #     resolution_parameter=resolution_parameter
        # )  # n_iterations=-1 to converge on best local result
        modScore = partition.modularity

        modInfo.append([modScore, partition.sizes()])
        modInfoScores[modScore].append(rNum)
        #  Output cluster membership and store for generating co-asociation matrix
        infoPerClustering.append(outputModules(outFoldT, outFoldLS, partition, runNum=rNum))
        #  Inform user of position
        count += 1
        if count == printE:
            printE = printE + tell
            #  Give some feedback about quality of results as working
            print(
                "\t\t#",
                count,
                " ModScore:",
                round(modScore, 3),
                " ModNum:",
                len(partition.sizes()),
                " ModSizes:",
                partition.sizes(),
            )

    #  Print out information on mod scores/sizes
    modInfoF = open(os.path.join(wF, baseFold, "moduleInfoAll.txt"), "w")
    print("Mod#\tModularityScore\tAvgModSize\tModuleNum\tModuleSizes", file=modInfoF)
    for i, inf in enumerate(modInfo):
        print(i + 1, inf[0], sum(inf[1]) / float(len(inf[1])), len(inf[1]), inf[1], sep="\t", file=modInfoF)
    modInfoF.close()

    if runNum > 1:
        ###########################################################################################
        #  Copy best results (based on modularity score) to bestFold folder.
        #  Create output folder
        outFoldTB = os.path.join(wF, baseFold, bestFold, outFoldTorig)
        outFoldLB = os.path.join(wF, baseFold, bestFold, outFoldLorig)
        if not os.path.exists(outFoldTB):
            os.makedirs(outFoldTB)
        if not os.path.exists(outFoldLB):
            os.makedirs(outFoldLB)

        infoPerClusteringBest = []
        copyNumber = int(round(runNum * (bestPerc / 100.0), 0))
        if copyNumber == 0:
            copyNumber = 1

        print(
            "\tWill copy Best ",
            bestPerc,
            "% (n=",
            copyNumber,
            ") results to ",
            os.path.join(baseFold, bestFold),
            sep="",
        )
        copyNum = 0

        modInfoF = open(os.path.join(wF, baseFold, "moduleInfoBest.txt"), "w")
        print("Mod#\tModularityScore\tAvgModSize\tModuleNum\tModuleSizes", file=modInfoF)
        bestScores = defaultdict(dict)
        for mScore in sorted(modInfoScores, reverse=True):
            if copyNum >= copyNumber:
                break
            for clustNum in modInfoScores[mScore]:
                if copyNum >= copyNumber:
                    break

                inf = modInfo[clustNum - 1]
                #  Store best info per cluster info (original)
                print(clustNum, inf[0], sum(inf[1]) / float(len(inf[1])), len(inf[1]), inf[1], sep="\t", file=modInfoF)
                infoPerClusteringBest.append(infoPerClustering[clustNum - 1])

                #  Write best module information for stats (VU)
                # the len will give the ranking of the scoring
                bestScores[len(infoPerClusteringBest)]["data"] = infoPerClusteringBest[-1]
                bestScores[len(infoPerClusteringBest)]["stats"] = {
                    "clustNum": clustNum,
                    "ModularityScore": inf[0],
                    "AvgModSize": sum(inf[1]) / float(len(inf[1])),
                    "ModuleNum": len(inf[1]),
                    "ModuleSizes": inf[1],
                }

                #  Copy best results to bestFold
                textName = str(clustNum) + ".csv"
                shutil.copy(os.path.join(outFoldT, textName), os.path.join(outFoldTB, textName))
                listName = "Clust" + str(clustNum)
                shutil.copytree(os.path.join(outFoldL, listName), os.path.join(outFoldLB, listName))

                #  Increment number copied
                copyNum += 1

        modInfoF.close()

        # save the data - graph, meta (i.e. config of the exp)
        saveStats(g, meta, bestScores, tfList, statsF)

        return partition.graph.vs["name"], infoPerClusteringBest
    else:
        return partition.graph.vs["name"], infoPerClustering


###########################################################################################


def loadTF(dataF, filename="TF_names_v_1.01"):
    tf_list = []
    tf_path = "{}/{}.txt".format(dataF, filename)
    if os.path.exists(tf_path):
        tf_list = np.genfromtxt(fname=tf_path, delimiter="\t", skip_header=1, dtype="str")
    return tf_list


def main(args: PGCNAArgs, finishT="Finished!"):
    overviewPrint(args)

    if args.singleCorrL:
        singleCorrList = loadCorrGeneList(args.workFolder, args.singleCorrListF)
    else:
        singleCorrList = None

    # Load list of TF
    tfList = loadTF(args.dataF)

    #  Load meta-file detailing list of files to process
    metaInf = loadMeta(args.metaInf, args.dataF)

    #  Find genes present in each data-set
    genesToKeep, requireFileNum = getGenes(metaInf, splitBy=args.fileSep, geneFrac=args.geneFrac, retainF=args.retainF)

    #### for random_genes
    tfList = loadTF(args.dataF)
    if args.ctrl_tfPath != "":
        print("#### Loaded the Random TF genes from ", args.ctrl_tfPath)
        import pandas as pd

        random_genes = pd.read_csv(args.ctrl_tfPath, sep="\t")
        # override tfList
        tfList = list(random_genes["gene"])

    #  Generate correlation HDF5 files
    corrh5, genesM = generateCorrMatrix(
        args.OUT_FOLDER,
        metaInf,
        genesToKeep,
        requireFileNum,
        singleCorrList,
        geneFrac=args.geneFrac,
        retainF=args.retainF,
        corrMatF=args.corrMatF,
        corrMatFS=args.corrMatFS,
        usePearson=args.usePearson,
        corrChunk=args.corrChunk,
        splitBy=args.fileSep,
        keepBigFA=args.keepBigFA,
        singleCorr=args.singleCorr,
        singleCorrL=args.singleCorrL,
    )

    # Generate marker genes
    markerGenes = mk.createMarkerGenes(genesM)

    corrh5 = integrateMutations(dataF=args.dataF, corrh5=corrh5, genesM=genesM, keepBigF=args.keepBigF, mutMetric=args.mutMetric, modifier_type=args.modifier)

    #  Reduce edges
    edgesFP, nodesFP = reduceEdges(
        args.OUT_FOLDER,
        args.dataF,
        args.gephiF,
        corrh5,
        genesM,
        markerGenes=markerGenes,
        retainF=args.retainF,
        edgePG=args.edgePG,
        edgePTF=args.edgePTF,
        corrMatF=args.corrMatF + "_GMF" + str(requireFileNum),
        keepBigFA=args.keepBigFA,
        keepBigF=args.keepBigF,
        ignoreDuplicates=args.ignoreDuplicates,
        roundL=args.roundL,
        tfList=tfList,
    )

    # Meta info useful for later analysis
    save_meta = {
        "expName": args.expName,
        "retainF": args.retainF,
        "edgesPG": int(args.edgePG),
        "edgesTF": int(args.edgePTF),
        "genesKept": len(genesToKeep),
        # "resolution": args.resolution_parameter,
        # "tf_list": tfList,
        "filesUsed": list(metaInf.keys()),
    }

    #  Run Leidenalg
    if not args.noLeidenalg:
        names, infoPerClustering = runleidenalgF(
            args.OUT_FOLDER,
            args.statsF,
            edgesFP,
            runNum=args.laRunNum,
            bestPerc=args.laBestPerc,
            baseFold=args.lBaseFold,
            outFoldTorig=args.lClustTxtFold,
            outFoldLorig=args.lClustListFold,
            meta=save_meta,
            resolution_parameter=args.resolution_parameter,
            tfList=tfList,
        )

    else:
        print("\nSkipping Leidenalg community detection and downstream output as --noLeidenalg is set")
    print(finishT)


if __name__ == "__main__":
    main()
