import math
import operator
from math import log
def cal_entropy(dataset):
    n = len(dataset)
    label_count = {}
    for data in dataset:
        label = data[-1]
        if label_count.has_key(label):
            label_count[label] += 1
        else:
            label_count[label] = 1
    entropy = 0
    for label in label_count:
        prob = float(label_count[label])/n
        entropy -= prob*log(prob,2)
    return entropy
def mean(datasetlist):
    n=len(datasetlist)
    return sorted(datasetlist)[n/2]
def mean1(datasetlist):
    n=len(datasetlist)
    return sorted(datasetlist)[n/2]
def split_dataset(dataset,feature_index,features):
    dataset_less = []
    dataset_greater = []
    label_less = []
    label_greater = []
    datasets = []
    data11=[]
    for data in dataset:
        data11.append(data[feature_index])
       # print data
        datasets.append(data[0:14])
    if features[feature_index] in ['age', 'fnlwgt', 'education-num', 'capital-gain','capital-loss','hours-per-week']:#
        mean_value=mean1(data11)
    else:
        mean_value = mean(data11)
    for data in dataset:
            if data[feature_index] > mean_value:
                dataset_greater.append(data)
                label_greater.append(data[-1])
            else:
                dataset_less.append(data)
                label_less.append(data[-1])
    return dataset_less,dataset_greater,label_less,label_greater

def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*math.log(prob, 2)
    return shannonEnt

def CreateDataSet(train_file):
    dataset = []
    for index,line in enumerate(open(train_file,'rU').readlines()):
        line = line.strip()
        fea_and_label = line.split(',')
        #if fea_and_label[0 between 20 and 35]
            #feaandlabel[0] = "20-35"
        #dataset.append([float(fea_and_label[i]) for i in range(len(fea_and_label)-1)]+[fea_and_label[len(fea_and_label)-1]])
        dataset.append([fea_and_label[i] for i in range(len(fea_and_label)-1)]+[fea_and_label[len(fea_and_label)-1]])
    features = ['age','workclass','fnlwgt','education','education-num','marital-status',\
                'occupation','relationship','race','sex','capital-gain',\
                'capital-loss','hours-per-week','native-country']
    return dataset, features

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def cal_info_gain(dataset,feature_index,base_entropy,features):
    datasets = []
    data11=[]
    for data in dataset:
        data11.append(data[feature_index])
       # print data
        datasets.append(data[0:14])
    if features[feature_index] in ['age', 'fnlwgt', 'education-num', 'capital-gain','capital-loss','hours-per-week']:#
        mean_value=mean1(data11)
    else:
        mean_value = mean(data11)
    dataset_less = []
    dataset_greater = []
    for data in dataset:
        if data[feature_index] > mean_value:
            dataset_greater.append(data)
        else:
            dataset_less.append(data)
    condition_entropy = float(len(dataset_less))/len(dataset)*cal_entropy(dataset_less) + float(len(dataset_greater))/len(dataset)*cal_entropy(dataset_greater)
    return base_entropy - condition_entropy
def chooseBestFeatureToSplit(dataSet,features):
    numberFeatures = len(features)-1
    baseEntropy = calcShannonEnt(dataSet)
    #print baseEntropy
    bestInfoGain = 0.0;
    bestFeature = -1;
    for i in range(numberFeatures):
        #if features[i] in ['age', 'fnlwgt', 'education-num', 'hours-per-week']:
        if features[i] in ['age', 'fnlwgt', 'education-num', 'hours-per-week','capital-gain', 'capital-loss']:
            infoGain=cal_info_gain(dataSet,i,baseEntropy,features)
        elif features[i] in [ 'workclass',  'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']:
            #print "nikanqingchu!!!",features[i]
            featList = [example[i] for example in dataSet]
            #n=len(dataSet)
            #print(featList)
            #featList=[]
            #while(' ?' in featList):
                #featList.remove(' ?')
            #print type(featList),featList
            uniqueVals = set(featList)
            #print "This is unique Vals",(uniqueVals)
            newEntropy =0.0
            #print "This is uniqueVals",uniqueVals
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
        else:
            print features
            print "in split: the feature[i]:",features[i]
            infoGain=0
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    #print "best!!!",bestFeature,bestInfoGain
    return bestFeature,bestInfoGain
def majorityCnt(classList):
    large=0
    small=0
    for vote in classList:
        if vote == ' >50K':
            large+=1
        elif vote == ' <=50K':
            small+=1
    if large>small:
        return ' >50K'
    else:
        return ' <=50K'

def majorityCnt2(classList):
    classCount ={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]=1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels,features):
    #print "in creat tree, the classlist:"
    classList = [example[-1] for example in dataSet]
    #print classList
    if len(classList)==0:
        return ' <=50K'
    if classList.count(classList[0])==len(classList):
        #print "all the same",classList,classList[0]
        return classList[0]
    if len(dataSet[0])==1:
        #print "only one feature:",majorityCnt(classList),classList
        return majorityCnt(classList)
    #######################################
    #if len(labels)==1 and labels[0] in ['age', 'fnlwgt', 'education-num', 'hours-per-week','capital-gain', 'capital-loss'] :
        #print "111",classList,majorityCnt(classList)
       # return majorityCnt(classList)
    #########################################
    if len(labels)==0:
        #print "in this strange place,majority",classList,majorityCnt(classList)
        return majorityCnt(classList)
    #print "in creat tree",labels
    bestFeat,bestInfoGain = chooseBestFeatureToSplit(dataSet,labels)
    if bestInfoGain<0.025:
        return majorityCnt(classList)
    #if bestFeat==-1:
        #print labels[bestFeat],classList,majorityCnt(classList)
    #print "bestfeature",bestFeat

    bestFeatLabel = labels[bestFeat]
    #print bestFeatLabel
    myTree = {bestFeatLabel:{}}
    #if labels[bestFeat] in ['age', 'fnlwgt', 'education-num', 'hours-per-week']:
    if labels[bestFeat] in ['age', 'fnlwgt', 'education-num', 'hours-per-week','capital-gain', 'capital-loss']:
        #featValues = [example[bestFeat] for example in dataSet]
        #uniqueVals = set(featValues)
        dataset_less,dataset_greater,labels_less,labels_greater = split_dataset(dataSet,bestFeat,labels)
        del(labels[bestFeat])
        myTree[bestFeatLabel]['<='] = createTree(dataset_less,labels,features)
        myTree[bestFeatLabel]['>'] = createTree(dataset_greater,labels,features)
    elif labels[bestFeat] in [ 'workclass',  'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']:
        #a=features.index(bestFeatLabel)
        #print "index!!!!infeurea",a,bestFeatLabel,features
        #featValues = [example[a] for example in dataSet]
        #uniqueVals = set(featValues)
        if labels[bestFeat]=='workclass':
            uniqueVals=[' Private', ' Self-emp-not-inc', ' Self-emp-inc', ' Federal-gov', ' Local-gov', ' State-gov', ' Without-pay', ' Never-worked']
        elif labels[bestFeat]=='education':
            uniqueVals=[' Bachelors', ' Some-college', ' 11th', ' HS-grad', ' Prof-school', ' Assoc-acdm', ' Assoc-voc', ' 9th', ' 7th-8th', ' 12th', ' Masters', ' 1st-4th', ' 10th', ' Doctorate', ' 5th-6th', ' Preschool']
        elif labels[bestFeat]=='marital-status':
            uniqueVals=[' Married-civ-spouse', ' Divorced', ' Never-married', ' Separated', ' Widowed', ' Married-spouse-absent', ' Married-AF-spouse']
        elif labels[bestFeat]=='occupation':
            uniqueVals=[' Tech-support', ' Craft-repair', ' Other-service', ' Sales', ' Exec-managerial', ' Prof-specialty', ' Handlers-cleaners', ' Machine-op-inspct', ' Adm-clerical', ' Farming-fishing', ' Transport-moving', ' Priv-house-serv', ' Protective-serv', ' Armed-Forces']
        elif labels[bestFeat]=='relationship':
            uniqueVals=[' Wife', ' Own-child', ' Husband', ' Not-in-family', ' Other-relative', ' Unmarried']
        elif labels[bestFeat]=='race':
            uniqueVals=[' White', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other', ' Black']
        elif labels[bestFeat]=='sex':
            uniqueVals=[' Female', ' Male']
        elif labels[bestFeat]=='native-country':
            uniqueVals=[' United-States', ' Cambodia', ' England', ' Puerto-Rico', ' Canada', ' Germany', ' Outlying-US(Guam-USVI-etc)', ' India', ' Japan', ' Greece', ' South', ' China', ' Cuba', ' Iran', ' Honduras', ' Philippines', ' Italy', ' Poland', ' Jamaica', ' Vietnam', ' Mexico', ' Portugal', ' Ireland', ' France', ' Dominican-Republic', ' Laos', ' Ecuador', ' Taiwan', ' Haiti', ' Columbia', ' Hungary', ' Guatemala', ' Nicaragua', ' Scotland', ' Thailand', ' Yugoslavia', ' El-Salvador', ' Trinadad&Tobago', ' Peru', ' Hong', ' Holand-Netherlands']

       # print "when creat tree",labels[bestFeat],uniqueVals
        del(labels[bestFeat])
        #print "this is unique Vals",uniqueVals
        for value in uniqueVals:
            if value!=' >50K'and value!=' <=50K':
                subLabels = labels[:]
                myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels,features)
           # subLabels = labels[:]
           # myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
def get_most(train_dataset):
    most_value=[]
    for item in range(14):
        data22=[]
        for data in train_dataset:
            data22.append(data[item])
        a=majorityCnt(data22)
        most_value.append(a)
    return most_value
def get_means(train_dataset):
    mean_value=[]
    for item in range(14):
        data11=[]
        for data in train_dataset:
            data11.append(data[item])
        if item in [0,2,4,10,11,12]:
            mean_value.append(mean1(data11))
        else:
            mean_value.append(mean(data11))
    return mean_value

def classify(decesion_tree,labels,test_data,mean_values,most_values):
    #print labels
    first_fea = decesion_tree.keys()[0]
    if first_fea==' >50K'or first_fea==' <=50K':
        return first_fea
    #print "nizijikan",first_fea,labels
    fea_index = labels.index(first_fea)
    #print "ni daoshi zhengqi a ",test_data[fea_index]
    if first_fea in ['age', 'fnlwgt', 'education-num', 'hours-per-week','capital-gain', 'capital-loss']:
    #if first_fea in ['age', 'fnlwgt', 'education-num', 'hours-per-week','capital-gain', 'capital-loss']:#
        if test_data[fea_index]==' ?':
            test_data[fea_index]=mean_values[fea_index]
        if test_data[fea_index] <= mean_values[fea_index]:
            sub_tree = decesion_tree[first_fea]['<=']
            if type(sub_tree) == dict:
                return classify(sub_tree,labels,test_data,mean_values,most_values)
            else:
                return sub_tree
        else:
            sub_tree = decesion_tree[first_fea]['>']
            if type(sub_tree) == dict:
                return classify(sub_tree,labels,test_data,mean_values,most_values)
            else:
                return sub_tree
    else:
        if test_data[fea_index]==' ?':
            #print "the mean value when creat tree",,first_fea,mean_values[fea_index]
            test_data[fea_index]=mean_values[fea_index]

        if test_data[fea_index] in decesion_tree[first_fea].keys():
            #print "kkkkkkkkkkkkkkkao!!!"
            sub_tree = decesion_tree[first_fea][test_data[fea_index]]
        else:
            sub_tree=' <=50K'
            #test_data[fea_index]=most_values[fea_index]
            #sub_tree = decesion_tree[first_fea][test_data[fea_index]]

        #print sub_tree
        if type(sub_tree) == dict:
            return classify(sub_tree,labels,test_data,mean_values,most_values)
        else:
            return sub_tree




def run(myDat,labels,myTestDat):
    #print labels
    bestfeature=chooseBestFeatureToSplit(myDat,labels)
    print bestfeature
    decisiontree=createTree(myDat,labels,labels)
    print decisiontree
    mean_values = get_means(myDat)
    most_values = get_most(myDat)
    print "In run  the mean_values",mean_values
    n = len(myTestDat)
    labels3=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    i=1
    f=open('task1.out2','w')
    f.write('Id,Target\n')
    #f.write(',')
    #f.write('Target')
    #f.write('\n')
    for test_data in myTestDat:
        label = classify(decisiontree,labels3,test_data,mean_values,most_values)
        #f.write('%s,%s/n'),i,label

        f.write(str.strip(str(i))+str.strip(',')+str.strip(label)+'\n')
        #f.write(',')
        #f.write(label)
        #f.write('\n')
        i+=1
    f.close()
def run2(myDat,labels,myTestDat):
    #print "in run 2",labels
    bestfeature=chooseBestFeatureToSplit(myDat,labels)
    print bestfeature
    decisiontree=createTree(myDat,labels,labels)
    print decisiontree
    mean_values = get_means(myDat)
    most_values = get_most(myDat)
    print "In run  the mean_values",mean_values
    n = len(myTestDat)
    correct = 0
    labels3=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    for test_data in myTestDat:
        label = classify(decisiontree,labels3,test_data,mean_values,most_values)
        print label,test_data[-1]
        if label==test_data[-1]:
            correct+=1
        #print test_data
        #print label
    print n
    print correct
    print "accurate rate is : ",correct/float(n)
#############################################################
print "11111"
train_file = "test.gt"
test_file="test.pred"
myDat,labels = CreateDataSet(train_file)
myTestDat,labels=CreateDataSet(test_file)
print "22222"
print myDat
run2(myDat,labels,myTestDat)


