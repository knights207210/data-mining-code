# 
# Fall 2017, CSCI 4502/5502, Homework 2 
# 
# Based on historical quotes from http://www.nasdaq.com/quotes/
# Be sure to follow the exact instruction for the output format 
#

import argparse
import csv
import numpy as np

def normalization (fname, attr, normType):
    '''
    Input Parameters:
        fname: Name of the csv file contiaining historical quotes
        attr: The attribute to be normalized 
        normType: The type of normalization 
    Output:
        For each line of quotes in the input file, print the original value and normalized value of the specific attribute, separated by <TAB>  
    '''

    #TODO: Write code given the Input / Output Paramters.

    # define two arrays to store the attribute data and the normalized data, a counter
    attr_value = []
    attr_normalized = []
    numObj =0

    # read file first
    with open(fname) as file:        
        file_data=csv.reader(file)
        for line in file_data:
            if numObj==0:
                for i in range(1,6):        # define which attribute
                    if(attr==line[i]):
                        break
            else:
                attr_value.append(float(line[i]))
            numObj = numObj+1

   
    # if min-max nomalization method   
    if normType =='min_max':
        min_value = min(attr_value)
        max_value = max(attr_value)
        for i in range(0,len(attr_value)):
            norm_value = (attr_value[i]-min_value)*(1.0-0)/(max_value-min_value)+0
            attr_normalized.append(norm_value)

    # IF Z-SCORE normalization method
    elif normType=='z_score':
        mean_value = np.mean(attr_value)
        std_value = np.std(attr_value)
        for i in range(0,len(attr_value)):
            norm_value = (attr_value[i]-mean_value)/std_value
            attr_normalized.append(norm_value)

    # print
    print(" Attr val \t Norm val\n")
    for i in range(0, len(attr_value)):
        print (" %f \t %f\n" % (attr_value[i],attr_normalized[i]))
 
    return attr_normalized
def correlation (fname1, attr1, fname2, attr2):
    '''
    Input Parameters:
        fname1: name of the first csv file containing historical quotes
        attr1: The attribute to consider in the first csv file (fname1)
        fname2: name of the second csv file containing historical quotes
        attr2: The attribute to consider in the second csv file (fname2)
        
    Output:
        correlation coefficient between attr1 in fname1 and attr2 in fname2
    '''

    #TODO: Write code given the Input / Output Paramters.

    # define two arrays to store the attribute data1 and data2, a counter
    attr_value1 = []
    attr_value2 = []
    numObj1 =0
    numObj2 =0
    # read file1 
    with open(fname1) as file1:        
        file1_data=csv.reader(file1)
        for line1 in file1_data:
            if numObj1 == 0:
                for i in range(1,6):        # define which attribute
                    if(attr1==line1[i]):
                        break
            else:
                attr_value1.append(float(line1[i]))
            numObj1 = numObj1+1

    # read file2 
    with open(fname2) as file2:        
        file2_data=csv.reader(file2)
        for line2 in file2_data:
            if numObj2 == 0:
                for i in range(1,6):        # define which attribute
                    if(attr2==line2[i]):
                        break
            else:
                attr_value2.append(float(line2[i]))
            numObj2 = numObj2+1
    
    corrcoef=np.corrcoef(attr_value1,attr_value2)
    print(corrcoef)
    return corrcoef

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Mining HW2')
    parser.add_argument('-f1', type=str,
                            help="First csv file. Use only f1 for Q1.", 
                            required=True)
    parser.add_argument("-f2", type=str, 
                            help="Second csv file. Used for Q2.", 
                            required=False)
    parser.add_argument("-n", type=str, 
                            help="Type of Normalization. Select either min_max or z_score",
                            choices=['min_max','z_score'],
                            required=False)
    parser.add_argument("-a1", type=str, 
                            help="Type of Attribute for fname1. Select either open or high or low or close or volume",
                            choices=['open','high','low','close','volume'],
                            required=False)
    parser.add_argument("-a2", type=str, 
                            help="Type of Attribute for fname2. Select either open or high or low or close or volume",
                            choices=['open','high','low','close','volume'],
                            required=False)


    args = parser.parse_args()

    if ( args.n and args.a1 ):
        normalization(args.f1, args.a1, args.n)
    elif ( args.f2 and args.a1 and args.a2):
        correlation(args.f1, args.a1, args.f2, args.a2)
    else:
        print "Need input in the following form:\nHW2_PythonTemplate.py -f1 <filename1> -a1 <attribute> -n <normalizationType> or \nHW2_PythonTemplate.py -f1 <filename1> -a1 <attribute> -f2 <filename2> -a2 <attribute>"
