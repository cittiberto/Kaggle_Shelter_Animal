from __future__ import division

import csv
import os
import csv
import pickle

import numpy as np
import scipy as sp


def buildTest(path1, path2):
    """
    Read the file and returns its content
    :param path: path file
    :return: file content as list of list
    """
    with open(path1, 'rb') as f:
        reader = csv.reader(f)
        all_items_list = list(reader)

    with open(path2, 'rb') as f:
        reader = csv.reader(f)
        test_items = list(reader)

    not_found = 0
    filtered_list = []
    c1=0
    c2=0
    found = 0
    for item1 in test_items:
        if(c1>0):
            cur_date = item1[2].split(" ")[0].split("-")[1]+"/"+item1[2].split(" ")[0].split("-")[2]+"/"+item1[2].split(" ")[0].split("-")[0]+" "+item1[2].split(" ")[1]
            if(0<=int(item1[2].split(" ")[1].split(":")[0])<=11):
                cur_date += " AM"
            else:
                if(int(item1[2].split(" ")[1].split(":")[0]) == 12):
                    new_hour = 12
                else:
                    new_hour = int(item1[2].split(" ")[1].split(":")[0]) - 12
                    if(0<=new_hour<10):
                        new_hour = str(new_hour)
                        new_hour = "0"+new_hour
                cur_date = item1[2].split(" ")[0].split("-")[1]+"/"+item1[2].split(" ")[0].split("-")[2]+"/"+item1[2].split(" ")[0].split("-")[0]+" "+str(new_hour)+":"+item1[2].split(" ")[1].split(":")[1]+":"+item1[2].split(" ")[1].split(":")[2]
                cur_date += " PM"
            found=0
            for item2 in all_items_list:
                if(c2>0):
                    if(found==0 and (item2[1]==item1[1] or item2[1]=="*"+item1[1]) and item2[2]==cur_date and (item2[8]==item1[5] or (item2[8]== 'NULL' and item1[5]=="")) and item2[6]==item1[3] and item1[4]==item2[7] and item2[9]==item1[6] and item1[7]==item2[10]):
                        filtered_list.append(item2[1]+","+item2[4])
                        print len(filtered_list)
                        print item2
                        found=1
                c2 +=1

            if(found==0):
                not_found +=1
                filtered_list.append(item2[1]+",null")
                print len(filtered_list)
                print "******************"

        c1 +=1

    count = 1
    oo = 0
    with open('../Output/filtered.csv', 'wb') as f:
        for s in filtered_list:
            if(s.split(",")[1]=='Died'):
                f.write("0,1,0,0,0" + '\n')
            if(s.split(",")[1]=='Adoption'):
                f.write("1,0,0,0,0" + '\n')
            if(s.split(",")[1]=='Euthanasia'):
                f.write("0,0,1,0,0" + '\n')
            if(s.split(",")[1]=='Return to Owner'):
                f.write("0,0,0,1,0" + '\n')
            if(s.split(",")[1]=='Transfer'):
                f.write("0,0,0,0,1" + '\n')
            if(s.split(",")[1]=='null'):
                f.write("0,0,0,0,1" + '\n')
            count += 1

    print str(not_found)+"Not found"


def readCsv1(path):

    list_to_return = []

    with open(path, 'rb') as f:
        reader = csv.reader(f)
        list_oo = list(reader)

    c = 0
    for i in list_oo:
        new_list=[]
        if(c!=0):
            for a in i:
                new_list.append(float(a))
            list_to_return.append(new_list)
        c += 1

    return list_to_return

def readCsv2(path):

    list_to_return = []

    with open(path, 'rb') as f:
        reader = csv.reader(f)
        list_oo = list(reader)

    c = 0
    for i in list_oo:
        new_list=[]
        if(c!=0):
            i.remove(str(c))
            for a in i:
                new_list.append(float(a))
            list_to_return.append(new_list)
        c += 1

    return list_to_return

def MultiLogLoss(act, pred):

    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def main():
    #buildTest("../Input/Austin_Animal_Center_Outcomes.csv","../Input/test.csv")

    best_result = 0.2189
    correct_res = readCsv1("../Output/filtered.csv")
    predictions_res = readCsv2("../Output/xgboostSubmission.csv")

    scores = []
    for index in range(0, len(predictions_res)):
        result = MultiLogLoss(correct_res[index], predictions_res[index])
        scores.append(result)

    if(sum(scores) / len(scores)< best_result):
        print str(sum(scores) / len(scores))+" -> Better performance"
    else:
        print str(sum(scores) / len(scores))+" -> Worse performance"

    """
    correct_res = readCsv1("../Output/filtered.csv")
    predictions_res = readCsv2("../Output/69773.csv")

    scores = []
    for index in range(0, len(predictions_res)):
        result = MultiLogLoss(correct_res[index], predictions_res[index])
        scores.append(result)

    print("Kaggle: 0.69773, "+" locale: "+str(sum(scores) / len(scores))+", differenza "+str(0.69773-sum(scores) / len(scores))) # 0.0985725708595

    correct_res = readCsv1("../Output/filtered.csv")
    predictions_res = readCsv2("../Output/69777.csv")

    scores = []
    for index in range(0, len(predictions_res)):
        result = MultiLogLoss(correct_res[index], predictions_res[index])
        scores.append(result)

    print("Kaggle: 0.69777, "+" locale: "+str(sum(scores) / len(scores))+", differenza "+str(0.69777-sum(scores) / len(scores))) # 0.0985725708595

    correct_res = readCsv1("../Output/filtered.csv")
    predictions_res = readCsv2("../Output/69798.csv")

    scores = []
    for index in range(0, len(predictions_res)):
        result = MultiLogLoss(correct_res[index], predictions_res[index])
        scores.append(result)

    print("Kaggle: 0.69798, "+" locale: "+str(sum(scores) / len(scores))+", differenza "+str(0.69798-sum(scores) / len(scores))) # 0.0985725708595


    correct_res = readCsv1("../Output/filtered.csv")
    predictions_res = readCsv2("../Output/69809.csv")

    scores = []
    for index in range(0, len(predictions_res)):
        result = MultiLogLoss(correct_res[index], predictions_res[index])
        scores.append(result)

    print("Kaggle: 0.69809, "+" locale: "+str(sum(scores) / len(scores))+", differenza "+str(0.69809-sum(scores) / len(scores))) # 0.0985725708595

    correct_res = readCsv1("../Output/filtered.csv")
    predictions_res = readCsv2("../Output/69904.csv")

    scores = []
    for index in range(0, len(predictions_res)):
        result = MultiLogLoss(correct_res[index], predictions_res[index])
        scores.append(result)

    print("Kaggle: 0.69904, "+" locale: "+str(sum(scores) / len(scores))+", differenza "+str(0.69904-sum(scores) / len(scores))) # 0.0985725708595

    correct_res = readCsv1("../Output/filtered.csv")
    predictions_res = readCsv2("../Output/86643.csv")

    scores = []
    for index in range(0, len(predictions_res)):
        result = MultiLogLoss(correct_res[index], predictions_res[index])
        scores.append(result)

    print("Kaggle: 0.86643, "+" locale: "+str(sum(scores) / len(scores))+", differenza "+str(0.86643-sum(scores) / len(scores))) # 0.0985725708595
    """


if __name__ == '__main__':
    main()
