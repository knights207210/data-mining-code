# Fall 2017, CSCI 4502/5502: Data Mining 
# Homework 1
# Dataset: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction 

import argparse
import csv
import numpy as np
import math

def calculate( dataFile, ithAttr):
    """
    Input Parameters:
        dataFile: The dataset file.
        ithAttre: The ith attribute for which the various properties must be calculated.

    Default value of 0,infinity,-infinity are assigned to all the variables as required. 
    Objective of the function is to calculate:  N (number of objects), min, max, mean, standard deviation, Q1, median, Q3, IQR
    """

    numObj, minValue, maxValue, mean, stdev, Q1, median, Q3, IQR = [0,"inf","-inf",0,0,0,0,0,0]
    
    #YOUR TASK: Write code to assign the values to the respective variables.
    
    # define an array to store the ith attribute data and a counter
    row_data=[]
    # read file first
    with open(dataFile) as file:
        file_data = csv.reader(file)
    # define a loop to get data row by row and calculate the numObj as well
        for row in file_data:
            if numObj!=0:     #don't need the first row necause the row contains no value 
                row_data.append(float(row[ithAttr-1]))
            numObj=numObj+1
    
    # after read in the data, start calculating
    numObj = numObj-1
    minValue = min(row_data)
    maxValue = max(row_data)
    mean = np.mean(row_data)
    stdev = math.sqrt(np.var(row_data))
    Q1 = np.percentile(row_data,25)
    median = np.median(row_data)
    Q3 = np.percentile(row_data,75)
    IQR = Q3-Q1

    return numObj, minValue, maxValue, mean, stdev, Q1, median, Q3, IQR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Mining HW1')
    parser.add_argument('--i', type=int,
                            help="ith attribute of the dataset (2 <= i <= 29)", 
                            default=5,
                            choices=range(2,30),
                            required=True)
    parser.add_argument("--data", type=str, 
                            help="Location of the dataset file",
                            default="energydata_complete.csv", 
                            required=True)
    args = parser.parse_args()

    print ','.join(map(str,calculate(args.data,args.i)))
