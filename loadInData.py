#!/usr/bin/env python3
# this will read in allele frequencies for multiple 
# populations from the files 
# created by (or in the style of) the output of vcftools --freq
# 
# these files are tab delimited and contain a header row
# 
# the first four columns are: chromosome name, position, 
# number of alleles, and number of observations used to make the estimate
# of these only the first two are used. The other two columns are requried to be there
# but the values are ignored by this script
# the following columns (fifth and thereafter) contain an allele and its frequency
# in the format AlleleName:###
# 
# each population is represented by one file and all files must have 
# _the same loci in the same order_
# 

import numpy as np
import sys

# function to read in one locus (one line) from all files at once
# and make sure all files have the same locus
# @param files an array of file objects that are open for reading with
#   each object representing a population
# @value an array of dictionaries, one per pop in the order of files,
#   with each dictionary containing alleleName:frequency pairs
def readLocus(files):
    # process the first population
    line = files[0].readline()
    # detect if we are at the end of the file(s)
    if not line:
        return None
    line = line.rstrip().split("\t")
    chr = line[0]
    pos = line[1]
    # array of dictionaries - one for each pop
    # each dictionary has k:v of alleleName:frequency
    popAF = [{}]
    if len(line) < 5:
        raise Exception("input file with no alleles listed for chr " + chr + " " + pos)
    for i in range(4, len(line)):
        allele = line[i].split(":")
        popAF[0][allele[0]] = float(allele[1]) # alleleName : frequency

    # commented out b/c should not happen - input check for >= 2 populations in Main()
    # if len(files) == 1:
    #     return popAF
    
    # process the rest of the populations
    for i in range(1, len(files)):
        line = files[i].readline().rstrip().split("\t")
        # check that chr and pos are the same
        if chr != line[0] or pos != line[1]:
            raise Exception("order of chr and pos does not match in all input files")
        # add dictionary for this pop
        popAF += [{}]
        if len(line) < 5:
            raise Exception("input file with no alleles listed for chr " + chr + " " + pos)
        for j in range(4, len(line)):
            allele = line[j].split(":")
            popAF[i][allele[0]] = float(allele[1]) # alleleName : frequency
    
    return popAF

# function to calculate average on the fly
# @param mean previously calculated mean
# @param newValue new value to add to the average
# @param n number of values total INCLUDING THE NEW VALUE
def updateAverage(mean, newValue, n):
    return mean + ((newValue - mean) / n)

def Main():

    # defaults
    inputFilePaths = [] # -f input allele frequency file paths (space separated)
    allowableFlags = ["-f"] # all possible flags, used for detecting end of file paths

    # get command line inputs
    flag = 1
    while flag < len(sys.argv):    #start at 1 b/c 0 is name of script
        if sys.argv[flag] == "-f":
            flag += 1
            while flag < len(sys.argv):
                if sys.argv[flag] in allowableFlags: # note this will cause an error if file path begins with "-"
                    break
                inputFilePaths += [sys.argv[flag]]
                flag += 1
            flag -= 1 # flag will be over-advanced by 1 after looping through all input files
        else:
            raise Exception("Error: option " + sys.argv[flag] + " not recognized.")
        flag += 1
    # end while loop for command line arguments

    # input error check(s)
    if len(inputFilePaths) < 2:
        raise Exception("At least two input files must be specified with -f")

    # set up memory for coancestry matrix
    # square matrix order = number of pops
    coanMatrix = np.zeros((len(inputFilePaths), len(inputFilePaths)))


    # open all population allele frequency files
    inputFreqFiles = [open(x, "r") for x in inputFilePaths]
    # skip header line
    for i in range(0, len(inputFreqFiles)):
        next(inputFreqFiles[i])

    # when working with potentially large genomic datasets (millions of loci)
    # it is often helpful (reduce memory demand) to parse locus by locus rather than 
    # reading the whole dataset into memory
    numLoci = 0 # number of loci used
    # for i in range(0,30): # for testing
    while True:
        # get allele frequencies
        locusAF = readLocus(inputFreqFiles)
        
        # break when at the end of the file(s)
        if locusAF is None:
            break

        numLoci += 1

        # get all allele names
        alleleNames = []
        for j in range(0, len(locusAF)):
            alleleNames += list(locusAF[j].keys())
        alleleNames = set(alleleNames) # remove duplicates

        # NOTE: this section of calculating coancestries
        # could be more efficient if we can have readLocus efficiently
        # return an np.array of allele frequencies (ordered) by pop 
        # allowing us to use np.array multiplication. If this section 
        # is a significant resource use, revise.
        #
        # now calculate coancestries as expected HOMOzygosity
        # note that expected HOMOzygosity = 1 - expected HETEROzygosity
        # for all pairs of pops including self comparisons
        # and update average coancestry matrix
        for j in range(0, len(locusAF)):
            for k in range(j, len(locusAF)):
                tempExpecHom = 0
                for a in alleleNames:
                    # expectHom is probability the same allele is sampled from each population
                    tempExpecHom += locusAF[j].get(a,0) * locusAF[k].get(a,0)
                coanMatrix[j,k] = updateAverage(coanMatrix[j,k], tempExpecHom, numLoci)
    # end of while loop calculating coancestries

    # make matrix symmetric
    # b/c crossing pop1 x pop2 = pop2 x pop1
    for i in range(0, coanMatrix.shape[0] - 1):
        for j in range(i + 1, coanMatrix.shape[0]):
            coanMatrix[j,i] = coanMatrix[i,j]

    # now we have a coancestry matrix that
    # is also a matrix whose values represent the 
    # (mean) expected HOMOzygosity of an offspring produced
    # by crossing individuals from two pops
    # NOTE: this can be considered the "M" matrix described in section 2.1 of Eding et al 2002
    # where they use the term "kinships" instead of coancestries
    print(coanMatrix)

    # an example of calculating average expected HOMOzygosity
    # (which is 1 - expected HETEROzygosity) for a population
    # created by selecting broodstock from the populations
    # with proportions given by c
    # NOTE: this is application of equation (1) from Eding et al 2002
    # for this example we are sampling broodstock in equal proportions from all populations
    # making this a two-dimensional array (matrix) here just to make the math explicitly match
    # Eding et al 2002 and related literature for ease of interpretation
    c = np.full(coanMatrix.shape[0], 1 / coanMatrix.shape[0])[:,None]
    meanOffspCoan = c.T @ coanMatrix @ c
    print(meanOffspCoan[0,0])

    # so the goal is to find c that MINIMIZES meanOffspCoan (which thereby 
    # MAXIMIZES 1 - meanOffspCoan, or expected heterozygosity) with a 
    # restricted number of values in c > 0
    # c is the proportion of broodstock sampled from each population,
    # so naturally the values of c are constrained to be in the range [0,1]
    # and sum(c) = 1



if __name__ == "__main__":
    Main()
