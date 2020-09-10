import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
import graphviz
import pydotplus
from IPython.display import Image

#print(dataset.head) # printing first few rows of the dataset
#print(dataset[' workclass'].describe())

#df = dataset[(dataset.astype(str) != ' ?').all(axis=1)]
#print(len(df))




def reading_data(filename):
    data=pd.read_csv(filename,sep=',') # reading the file
    data.columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
    print("length of dataset before cleaning", len(data))
    return data

def income(text):
    if '50K' in text:
        return 1
    else:
        return 0


def cleaning_data(data):
    print("length of dataset before cleaning", len(data))
    data = data[(data.astype(str) != ' ?').all(axis=1)] # removing rows that have '?' in it.
    print("length of dataset after cleaning",len(data))
    data=data.drop_duplicates() # dropping duplicate rows
    print("length of dataset after removing duplicates", len(data))
    return data





def preprocessing_data(data):

    data['salary'] = data['salary'].astype('category') # changing '>50k' and '<=50' to categories

    data = data.drop(['capital-gain', 'capital-loss', 'native-country', 'fnlwgt'], axis=1) # dropping irrelevant columns

    data=pd.get_dummies(data,columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']) # changing categorical values to one hot encoding

    return data


def split_train_test(data):
    data_train = data[:23000]
    data_test = data[23000:]
    d_train_data = data_train.drop(['salary'], axis=1)
    d_train_binary = data_train['salary']
    d_test_data = data_test.drop(['salary'], axis=1)
    d_test_binary = data_test['salary']
    print(d_train_data)
    print(d_train_binary)

    return d_train_data, d_train_binary



def main():
    dataset=reading_data('adult.csv')
    print(dataset)
    #print(dataset['age'].describe())
    dataset=cleaning_data(dataset)
    dataset=preprocessing_data(dataset)
    print("length of data set" ,len(dataset))

    # training
    train_attributes, train_binary = split_train_test(dataset)

    t=tree.DecisionTreeClassifier(criterion='entropy',max_depth=7)
    t=t.fit(train_attributes,train_binary)

    # visualize tree
    dot_data = tree.export_graphviz(t, out_file=None, label='all', impurity=False, proportion=True,feature_names=list(train_attributes), class_names=['lt50K', 'gt50K'],
                                    filled=True, rounded=True)
    graph=pydotplus.graph_from_dot_data(dot_data)

    #print(graph)
    # graph.write_png('tree.png')
    graph = graphviz.Source(graph)
    graph.render("tree", view=True)




if __name__ == '__main__':
    main()