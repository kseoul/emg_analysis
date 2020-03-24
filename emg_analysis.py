import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#gtdir = r'C:\Users\J9857007\Desktop\data\groundTruth'
#myodir = r'C:\Users\J9857007\Desktop\data\MyoData'
gtdir = r'C:\Users\J\Desktop\Data_Mining_Assign1Data\groundTruth'
myodir = r'C:\Users\J\Desktop\Data_Mining_Assign1Data\MyoData'


# These files are in myodir
vidinfo = "_video_info.csv"
emg = "_EMG.txt"
imu = "_IMU.txt"
dfcat = ('fork','spoon')
emg_col = ['ts','emg1','emg2','emg3','emg4','emg5','emg6','emg7','emg8']
gt_col = ['start','stop','x']


ls = []

# Get each user and filenames for each user by fork and spoon
#userls.append(root[len(gtdir):].split("\\",2)[1])
#ls.append(os.path.splitext(file)[0])
for root, dirs, files in os.walk(gtdir):
    for file in files:
        if file.endswith(".txt"):
            userdir = root[len(gtdir)+1:]
            # Append to a list information for each user, video info, and EMG filepath
            ls.append([root[len(gtdir):].split("\\",2)[1], #[0] User e.g User09
                       root[len(gtdir):].split("\\",2)[2], #[1] Type e.g Spoon, Fork
                       os.path.splitext(file)[0],#[2] filename e.g 1503512024751
                       os.path.join(root, file), #[3] gt file e.g C:\Users\J9857007\Desktop\data\groundTruth\user10\fork\1503512024740.txt
                       root[len(gtdir):], #[4] User dir e.g \user10\fork
                       os.path.join(myodir,userdir,os.path.splitext(file)[0] + emg), #[5] myo emg data e.g ..\MyoData\user10\fork\1503512024740_EMG.txt
                       os.path.join(myodir,userdir,os.path.splitext(file)[0] + vidinfo), #[6] VideoInfo File
                       pd.read_csv(os.path.join(myodir,userdir,os.path.splitext(file)[0] + vidinfo),header=None)[0][0], #[7] fps info e.g. 29.884
                       pd.read_csv(os.path.join(myodir,userdir,os.path.splitext(file)[0] + vidinfo),header=None)[1][0] # [8] video last frame info
                       ])

# Defining function to categorize each frame as eating or non-eating based on ground truth
def ts_cat(time_data, main_data, fps, lf, tf):
    df1 = pd.DataFrame(columns = main_data.columns)
    df2 = df1
    for index in time_data.index:
        start = ((lf - time_data['start'][index]) / fps * 1000).astype(int)
        stop = ((lf - time_data['stop'][index]) / fps * 1000).astype(int)
        print(start, stop)
        df1 = df1.append(main_data[(main_data.index >= tf - start) & (main_data.index <= tf - stop)])
    df2 = main_data[~main_data.isin(df1.index.duplicated(keep='first'))].dropna()
    df1['status'] = 'eating'
    df2['status'] = 'non-eating'
    #print(df1)
    df1 = df1.append(df2)
    return df1

# Creating a dataframe with all user data
raw_df = pd.DataFrame(columns = emg_col + ['user','action']).drop(columns=['ts'])
for cnt in range(len(ls)):
    print(cnt, ls[cnt][0])
    dfgt = pd.read_csv(ls[cnt][3], names=gt_col)[::-1]
    dfemg = pd.read_csv(ls[cnt][5], names=emg_col, index_col=0)
    fs = ls[cnt][7]
    lf = ls[cnt][8]
    tf = dfemg.index[-1]
    dfx = ts_cat(dfgt,dfemg,fs,lf,tf)
    dfx['user'] = ls[cnt][0]
    dfx['action'] = ls[cnt][1]
    raw_df = raw_df.append(dfx)

# A function to build feature matrix (mean, min max, std, rsm)
def build_matrix():
    cols = ['id', 'emg1','emg2','emg3','emg4','emg5','emg6','emg7','emg8']
    cols.pop(0)
    features = ['mean','min','max','std','rms']
    ls = []
    for i in cols:
        for j in features:
            ls.append(i+"_"+j)
    return ls

# A function to extract features (mean, min, max, std, rsm) from the dataframe
def feature_extraction(df, steps):
    feature_matrix = build_matrix()
    feature_matrix = pd.DataFrame(columns = feature_matrix)
    feature_matrix.shape
    step = 0
    for loop in range(np.floor(len(df)/steps).astype(int)): 
        ls = []
        for i in df:
            ls.append(df[i][step:step+steps].mean())
            ls.append(df[i][step:step+steps].min())
            ls.append(df[i][step:step+steps].max())
            ls.append(df[i][step:step+steps].std())
            ls.append(np.sqrt(np.mean(df[i][step:step+steps]**2)))#rsm
        step += steps
        feature_matrix.loc[loop] = ls
    feature_matrix = feature_matrix.apply(pd.to_numeric)
    return feature_matrix

# Create list of users
users = raw_df['user'].drop_duplicates().values
fmatrix = build_matrix()

# Create a dictionary by each user along with eating & non eating activities
newls = {}
for i in users:
    this_df = raw_df[raw_df['user'] == i]
    eat = this_df[this_df['status']=='eating'].drop(columns = ['action','status','user'])
    neat = this_df[this_df['status']=='non-eating'].drop(columns = ['action','status','user'])
    print(i, len(eat), len(neat))
    if len(eat) >= len(neat):
        x = len(neat)
    else:
        x = len(eat)
    neat = neat.sample(n=x)
    newls[i] = {'eat' : eat, 
         'neat' : neat}
    this_df = []

# Aggregate total user data and put into total_df dataframe
total_df = pd.DataFrame(columns = build_matrix() + ['user', 'status'])
for user in users:    
    fneat = feature_extraction(newls[user]['neat'], 100)
    fneat['status'] = 'non-eating'
    fneat['user'] = user
    feat = feature_extraction(newls[user]['eat'], 100)
    feat['status'] = 'eating'
    feat['user'] = user
    total_df = total_df.append(fneat)
    total_df = total_df.append(feat)
    print(user, " added to total df" )
###########################################
#Phase One
###########################################
# Perform Feature Selection, PCA, and machine learning model on each user (Phase ONE)
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Dense

sc = StandardScaler()
user_results = {}
for user in users:
    array = total_df[total_df['user'] == user].values
    arrayX = array[:,0:40]
    arrayStat = array[:,40:42]
    X_train = arrayX
    pca = PCA(n_components = 5)
    scaled_X = sc.fit_transform(X_train)
    scaled_eat = pd.DataFrame(scaled_X, columns=build_matrix())
    pca.fit(scaled_eat)
    principalComponents = pca.transform(scaled_eat)
    principalComponents    
    pca_df = pd.DataFrame(principalComponents, columns = ['PCA1','PCA2','PCA3','PCA4','PCA5'])
    pca_df['status'] = arrayStat[:,0]
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PCA1', fontsize = 10)
    ax.set_ylabel('PCA2', fontsize = 10)
    ax.set_title(user + " 2 component PCA", fontsize = 20)
    type_class = ['non-eating','eating']
    colors = ['b','y']
    
    for type_classes, color in zip(type_class,colors):
        keepIndices = pca_df['status']==type_classes
        ax.scatter(pca_df.loc[keepIndices, 'PCA1'], pca_df.loc[keepIndices,'PCA2'], c = color, s=10)
    plt.legend(type_class)
    plt.grid()
    plt.savefig(user+".png")
    plt.show()
    
    # Decision Tree Model
    X_train, X_test, y_train, y_test = train_test_split(pca_df[['PCA1','PCA2','PCA3']].values, pca_df['status'].values, test_size=0.4, random_state=1) # 60% training and 40% test
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Decision Tree Model Results")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    precision = precision_score(y_test,y_pred,pos_label='eating')
    print("Precision: ", precision)
    recall = recall_score(y_test,y_pred,pos_label='eating')
    print("Recall: ", recall)
    f1 = f1_score(y_test,y_pred,pos_label='eating')
    print("F1: ", f1)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    dtcresults = {"Accuracy" : accuracy, "Precision" : precision, "Recall": recall, "F1" : f1}
    
    #SVM Model
    X_train, X_test, y_train, y_test = train_test_split(pca_df[['PCA1','PCA2','PCA3']].values, pca_df['status'].values, test_size=0.4, random_state=1) # 60% training and 40% test
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test) 
    print("SVM Model Results")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    precision = precision_score(y_test,y_pred,pos_label='eating')
    print("Precision: ", precision)
    recall = recall_score(y_test,y_pred,pos_label='eating')
    print("Recall: ", recall)
    f1 = f1_score(y_test,y_pred,pos_label='eating')
    print("F1: ", f1)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    svmresults = {"Accuracy" : accuracy, "Precision" : precision, "Recall": recall, "F1" : f1}
    
    
    #Neural Network
    
    X = sc.fit_transform(pca_df[['PCA1','PCA2','PCA3']].values)
    y = pca_df['status'].values
    
    onehotencoder = OneHotEncoder()
    y = y.reshape(-1,1)
    y = onehotencoder.fit_transform(y).toarray()
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) # 60% training and 40% test
    
    model = Sequential()
    model.add(Dense(16, input_dim=3, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
     
    history = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=100, batch_size=64)
    y_pred = model.predict(X_test)
    #Converting predictions to label
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    #Converting one hot encoded test label to label
    test = list()
    for i in range(len(y_test)):
        test.append(np.argmax(y_test[i]))
    
    from sklearn.metrics import accuracy_score
    a = accuracy_score(pred,test)
    print('Accuracy is:', a*100)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.show()
    
    print("Neural Network Results")
    accuracy = accuracy_score(test, pred)
    print("Accuracy: ", accuracy)
    precision = precision_score(test,pred)
    print("Precision: ", precision)
    recall = recall_score(test,pred)
    print("Recall: ", recall)
    f1 = f1_score(test,pred)
    print("F1: ", f1)
    #print(confusion_matrix(y_test,y_pred))
    nnresults = {"Accuracy" : accuracy, "Precision" : precision, "Recall": recall, "F1" : f1}
    user_results[user] = {'DT': dtcresults, 'SVM' : svmresults, 'NN' : nnresults}

#Create df to export as csv file called results.csv
data = [user_results]
user_ls = []
for i in data:
    for key in i:
        user_ls.append([key, 'DT', 'Accuracy', user_results[key]['DT']['Accuracy']])
        user_ls.append([key, 'DT', 'Precision', user_results[key]['DT']['Precision']])
        user_ls.append([key, 'DT', 'Recall', user_results[key]['DT']['Recall']])
        user_ls.append([key, 'DT', 'F1', user_results[key]['DT']['F1']])
        user_ls.append([key, 'SVM', 'Accuracy', user_results[key]['SVM']['Accuracy']])
        user_ls.append([key, 'SVM', 'Precision', user_results[key]['SVM']['Precision']])
        user_ls.append([key, 'SVM', 'Recall', user_results[key]['SVM']['Recall']])
        user_ls.append([key, 'SVM', 'F1', user_results[key]['SVM']['F1']])
        user_ls.append([key, 'NN', 'Accuracy', user_results[key]['NN']['Accuracy']])
        user_ls.append([key, 'NN', 'Precision', user_results[key]['NN']['Precision']])
        user_ls.append([key, 'NN', 'Recall', user_results[key]['NN']['Recall']])
        user_ls.append([key, 'NN', 'F1', user_results[key]['NN']['F1']])
pd.DataFrame(user_ls, columns = ['user','model','metric','value']).to_csv('results.csv')


###########################################
#Phase Two
###########################################

# Run feature extraction and models on all data independent of users (PHASE 2)
for user in range(1):
    array = total_df.values
    arrayX = array[:,0:40]
    arrayStat = array[:,40:42]
    X_train = arrayX
    pca = PCA(n_components = 5)
    scaled_X = sc.fit_transform(X_train)
    scaled_eat = pd.DataFrame(scaled_X, columns=build_matrix())
    pca.fit(scaled_eat)
    principalComponents = pca.transform(scaled_eat)
    principalComponents    
    pca_df = pd.DataFrame(principalComponents, columns = ['PCA1','PCA2','PCA3','PCA4','PCA5'])
    pca_df['status'] = arrayStat[:,0]
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PCA1', fontsize = 10)
    ax.set_ylabel('PCA2', fontsize = 10)
    ax.set_title("All data 2 component PCA", fontsize = 20)
    type_class = ['non-eating','eating']
    colors = ['b','y']
    
    for type_classes, color in zip(type_class,colors):
        keepIndices = pca_df['status']==type_classes
        ax.scatter(pca_df.loc[keepIndices, 'PCA1'], pca_df.loc[keepIndices,'PCA2'], c = color, s=10)
    plt.legend(type_class)
    plt.grid()
    plt.savefig("All data PCA.png")
    plt.show()
    
    # Decision Tree Model
    X_train, X_test, y_train, y_test = train_test_split(pca_df[['PCA1','PCA2','PCA3']].values, pca_df['status'].values, test_size=0.4, random_state=1) # 60% training and 40% test
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("Decision Tree Model Results")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    precision = precision_score(y_test,y_pred,pos_label='eating')
    print("Precision: ", precision)
    recall = recall_score(y_test,y_pred,pos_label='eating')
    print("Recall: ", recall)
    f1 = f1_score(y_test,y_pred,pos_label='eating')
    print("F1: ", f1)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    dtcresults = {"Accuracy" : accuracy, "Precision" : precision, "Recall": recall, "F1" : f1}
    
    #SVM Model
    X_train, X_test, y_train, y_test = train_test_split(pca_df[['PCA1','PCA2','PCA3']].values, pca_df['status'].values, test_size=0.4, random_state=1) # 60% training and 40% test
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test) 
    print("SVM Model Results")
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    precision = precision_score(y_test,y_pred,pos_label='eating')
    print("Precision: ", precision)
    recall = recall_score(y_test,y_pred,pos_label='eating')
    print("Recall: ", recall)
    f1 = f1_score(y_test,y_pred,pos_label='eating')
    print("F1: ", f1)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    svmresults = {"Accuracy" : accuracy, "Precision" : precision, "Recall": recall, "F1" : f1}
    
    
    #Neural Network    
    X = sc.fit_transform(pca_df[['PCA1','PCA2','PCA3']].values)
    y = pca_df['status'].values
    onehotencoder = OneHotEncoder()
    y = y.reshape(-1,1)
    y = onehotencoder.fit_transform(y).toarray() 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) # 60% training and 40% test
    
    model = Sequential()
    model.add(Dense(16, input_dim=3, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=100, batch_size=64)
    y_pred = model.predict(X_test)
    #Converting predictions to label
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    #Converting one hot encoded test label to label
    test = list()
    for i in range(len(y_test)):
        test.append(np.argmax(y_test[i]))
    
    from sklearn.metrics import accuracy_score
    a = accuracy_score(pred,test)
    print('Accuracy is:', a*100)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.show()
    
    print("Neural Network Results")
    accuracy = accuracy_score(test, pred)
    print("Accuracy: ", accuracy)
    precision = precision_score(test,pred)
    print("Precision: ", precision)
    recall = recall_score(test,pred)
    print("Recall: ", recall)
    f1 = f1_score(test,pred)
    print("F1: ", f1)
    #print(confusion_matrix(y_test,y_pred))
    nnresults = {"Accuracy" : accuracy, "Precision" : precision, "Recall": recall, "F1" : f1}
    user_results['all'] = {'DT': dtcresults, 'SVM' : svmresults, 'NN' : nnresults}

# Append to user ls df and dump to 'all results.csv' as needed
user_ls.append(['all', 'DT', 'Accuracy', user_results['all']['DT']['Accuracy']])
user_ls.append(['all', 'DT', 'Precision', user_results['all']['DT']['Precision']])
user_ls.append(['all', 'DT', 'Recall', user_results['all']['DT']['Recall']])
user_ls.append(['all', 'DT', 'F1', user_results['all']['DT']['F1']])
user_ls.append(['all', 'SVM', 'Accuracy', user_results['all']['SVM']['Accuracy']])
user_ls.append(['all', 'SVM', 'Precision', user_results['all']['SVM']['Precision']])
user_ls.append(['all', 'SVM', 'Recall', user_results['all']['SVM']['Recall']])
user_ls.append(['all', 'SVM', 'F1', user_results['all']['SVM']['F1']])
user_ls.append(['all', 'NN', 'Accuracy', user_results['all']['NN']['Accuracy']])
user_ls.append(['all', 'NN', 'Precision', user_results['all']['NN']['Precision']])
user_ls.append(['all', 'NN', 'Recall', user_results['all']['NN']['Recall']])
user_ls.append(['all', 'NN', 'F1', user_results['all']['NN']['F1']])

pd.DataFrame(user_ls).to_csv('all results.csv')
