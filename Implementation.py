"""
Created on Tue Jan 11 15:13:00 2022

@author: Nikola
"""
#%% 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay, precision_score, f1_score, ConfusionMatrixDisplay
import matplotlib
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics


#%% 

df = pd.read_csv("C:/Users/Nikola/Desktop/Domaci2 OBLICI/recipes.csv")

df.shape
df.head()
df.info()
obelezja = df.columns

#%% 
df.drop('Unnamed: 0', inplace= True, axis = 1)

countryGroupBy = df.groupby("country").sum()
drzave = df.iloc[:,-1].unique()

#%% 
oil_cols = [col for col in df.columns if 'oil' in col]

for i in oil_cols:
    plt.figure(figsize=(9, 8))
    plt.plot(countryGroupBy[i])
    plt.title(i)
    
    
#%%
milk = [col for col in df.columns if 'milk' in col]
juice = [col for col in df.columns if 'juice' in col]
sugar = [col for col in df.columns if 'sugar' in col]
oil = [col for col in df.columns if 'oil' in col]
onion = [col for col in df.columns if 'onion' in col]
chili = [col for col in df.columns if 'chili' in col]

milk
juice
sugar
#%%

n=9
arange = np.arange(n)
width = 0.10

plt.figure(figsize=(8,6))
plt.bar(arange - 0.1  , countryGroupBy['milk'], color = 'red',width = width, edgecolor = 'black', label='milk')
plt.bar(arange      , countryGroupBy['buttermilk'], color = 'blue',width = width, edgecolor = 'black',label='buttermilk')
plt.bar(arange + 0.1, countryGroupBy['coconut milk'], color = 'green',width = width, edgecolor = 'black', label='coconut milk')     
plt.xlabel("Drzave")
plt.ylabel("Kolicina mlecnih sastojaka")
plt.title("Kolicina mlecnih sastojaka po drzavi")
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], [drzave[0], drzave[1], drzave[2], drzave[3], drzave[4], drzave[5],drzave[6], drzave[7], drzave[8]])
plt.legend()
plt.xticks(rotation=45)
plt.show()
    

#%%
    


n=9
arange = np.arange(n)
width = 0.10

plt.figure(figsize=(8,6))
plt.bar(arange - 0.1  , countryGroupBy['chili'], color = 'red',width = width, edgecolor = 'black', label='chili')
plt.bar(arange      , countryGroupBy['chili powder'], color = 'blue',width = width, edgecolor = 'black',label='chili powder')
plt.bar(arange + 0.1, countryGroupBy['jalapeno chilies'], color = 'green',width = width, edgecolor = 'black', label='jalapeno chilies')     
plt.xlabel("Drzave")
plt.ylabel("Kolicina ljutih sastojaka")
plt.title("Kolicina ljutih sastojaka po drzavi")
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], [drzave[0], drzave[1], drzave[2], drzave[3], drzave[4], drzave[5],drzave[6], drzave[7], drzave[8]])
plt.legend()
plt.xticks(rotation=45)
plt.show()

#%%  

n=9
arange = np.arange(n)
width = 0.10

plt.figure(figsize=(8,6))
plt.bar(arange - 0.1  , countryGroupBy['sugar'], color = 'red',width = width, edgecolor = 'black', label='sugar')
plt.bar(arange      , countryGroupBy['brown sugar'], color = 'blue',width = width, edgecolor = 'black',label='brown sugar')
plt.bar(arange + 0.1, countryGroupBy['white sugar'], color = 'green',width = width, edgecolor = 'black', label='white sugar')
plt.bar(arange + 0.2, countryGroupBy['granulated sugar'], color = 'green',width = width, edgecolor = 'black', label='granulated sugar')
plt.xlabel("Drzave")
plt.ylabel("Kolicina raznih vrsta secera")
plt.title("Kolicina raznih vrsta secera po drzavi")
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], [drzave[0], drzave[1], drzave[2], drzave[3], drzave[4], drzave[5],drzave[6], drzave[7], drzave[8]])
plt.legend()
plt.xticks(rotation=45)
plt.show()

#%%

french = df.loc[df['country']=='french'].drop(['country'], axis=1)
french_sum = french.sum(axis=0).sort_values(ascending=False)

lista=[]
k = 0
for i in french_sum:
    if i < 350:
        lista.append(k)
    k = k + 1
    
    
french = french.drop(french.columns[lista], axis=1)
french_sum = french_sum.drop(french_sum.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(french.columns,french_sum,color=['gray', 'red', 'green', 'blue', 'cyan',"orange"])
plt.xlabel("Sastojci")
plt.ylabel("Kolicina sastojaka")
plt.xticks(rotation=45)
plt.title("Francuska")
plt.show() 

#%%
southern_us = df.loc[df['country']=='southern_us'].drop(['country'], axis=1)
southern_us_sum = southern_us.sum(axis=0).sort_values(ascending=False)


lista=[]
k = 0
for i in southern_us_sum:
    if i < 500:
        lista.append(k)
    k = k + 1
    
    
southern_us = southern_us.drop(southern_us.columns[lista], axis=1)
southern_us_sum = southern_us_sum.drop(southern_us_sum.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(southern_us.columns,southern_us_sum, color=['gray', 'red', 'green', 'blue', 'cyan',"orange"])
plt.xlabel("Sastojci")
plt.ylabel("Kolicina sastojaka")
plt.xticks(rotation=45)
plt.title("Juzna amerika")
plt.show() 
#%%

greek = df.loc[df['country']=='greek'].drop(['country'], axis=1)
greek_sum = greek.sum(axis=0).sort_values(ascending=False)

lista=[]
k = 0
for i in greek_sum:
    if i < 200:
        lista.append(k)
    k = k + 1
    
    
greek = greek.drop(greek.columns[lista], axis=1)
greek_sum = greek_sum.drop(greek_sum.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(greek.columns,greek_sum, color=['gray', 'red', 'green', 'blue', 'cyan',"orange"])
plt.xlabel("Sastojci")
plt.ylabel("Kolicina sastojaka")
plt.xticks(rotation=45)
plt.title("Grcka")
plt.show() 
#%%

mexican = df.loc[df['country']=='mexican'].drop(['country'], axis=1)
mexican_sum = mexican.sum(axis=0).sort_values(ascending=False)

lista=[]
k = 0
for i in mexican_sum:
    if i < 350:
        lista.append(k)
    k = k + 1
    
drzave
mexican = mexican.drop(mexican.columns[lista], axis=1)
mexican_sum = mexican_sum.drop(mexican_sum.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(mexican.columns,mexican_sum,color=['gray', 'red', 'green', 'blue', 'cyan',"orange"])
plt.xlabel("Sastojci")
plt.ylabel("Kolicina sastojaka")
plt.xticks(rotation=45)
plt.title("Meksiko")
plt.show() 

#%%

italian = df.loc[df['country']=='italian'].drop(['country'], axis=1)
italian_sum = italian.sum(axis=0).sort_values(ascending=False)

lista=[]
k = 0
for i in italian_sum:
    if i < 350:
        lista.append(k)
    k = k + 1
    
drzave
italian = italian.drop(italian.columns[lista], axis=1)
italian_sum = italian_sum.drop(italian_sum.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(italian.columns,italian_sum,color=['gray', 'red', 'green', 'blue', 'cyan',"orange"])
plt.xlabel("Sastojci")
plt.ylabel("Kolicina sastojaka")
plt.xticks(rotation=45)
plt.title("Italija")
plt.show() 

#%%

japanese = df.loc[df['country']=='japanese'].drop(['country'], axis=1)
japanese_sum = japanese.sum(axis=0).sort_values(ascending=False)

lista=[]
k = 0
for i in japanese_sum:
    if i < 190:
        lista.append(k)
    k = k + 1
    
drzave
japanese = japanese.drop(japanese.columns[lista], axis=1)
japanese_sum = japanese_sum.drop(japanese_sum.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(japanese.columns,japanese_sum,color=['gray', 'red', 'green', 'blue', 'cyan',"orange"])
plt.xlabel("Sastojci")
plt.ylabel("Kolicina sastojaka")
plt.xticks(rotation=45)
plt.title("Japan")
plt.show() 

#%%

chinese = df.loc[df['country']=='chinese'].drop(['country'], axis=1)
chinese_sum = chinese.sum(axis=0).sort_values(ascending=False)

lista=[]
k = 0
for i in chinese_sum:
    if i < 450:
        lista.append(k)
    k = k + 1
    
drzave
chinese = chinese.drop(chinese.columns[lista], axis=1)
chinese_sum = chinese_sum.drop(chinese_sum.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(chinese.columns,chinese_sum,color=['gray', 'red', 'green', 'blue', 'cyan',"orange"])
plt.xlabel("Sastojci")
plt.ylabel("Kolicina sastojaka")
plt.xticks(rotation=45)
plt.title("Kina")
plt.show() 
#%%

thai = df.loc[df['country']=='thai'].drop(['country'], axis=1)
thai_sum = thai.sum(axis=0).sort_values(ascending=False)

lista=[]
k = 0
for i in thai_sum:
    if i < 210:
        lista.append(k)
    k = k + 1
    
drzave
thai = thai.drop(thai.columns[lista], axis=1)
thai_sum = thai_sum.drop(thai_sum.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(thai.columns,thai_sum,color=['gray', 'red', 'green', 'blue', 'cyan',"orange"])
plt.xlabel("Sastojci")
plt.ylabel("Kolicina sastojaka")
plt.xticks(rotation=45)
plt.title("Tajland")
plt.show() 

#%%

british = df.loc[df['country']=='british'].drop(['country'], axis=1)
british_sum = british.sum(axis=0).sort_values(ascending=False)

lista=[]
k = 0
for i in british_sum:
    if i < 120:
        lista.append(k)
    k = k + 1
    
drzave
british = british.drop(british.columns[lista], axis=1)
british_sum = british_sum.drop(british_sum.index[lista])
plt.figure(figsize=(5,5)).set_edgecolor('black')
plt.bar(british.columns,british_sum,color=['gray', 'red', 'green', 'blue', 'cyan',"orange"])
plt.xlabel("Sastojci")
plt.ylabel("Kolicina sastojaka")
plt.xticks(rotation=45)
plt.title("Velika Britanija")
plt.show()     

#%% 

X = df.iloc[:,:-1].copy()
y = df.iloc[:,-1].copy()

print(X.shape)
print(y.unique())
labels_y = y.unique()
print(labels_y)
X.head()
describe = X.describe()


#%%

print('nedostajućih vrednosti ima: ', df.isnull().sum().sum())




print('uzoraka u SOUTH_US klasi ima: ', sum(y=='southern_us'))
print('uzoraka u FRANCE klasi ima: ', sum(y=='french'))
print('uzoraka u GREECE klasi ima: ', sum(y=='greek'))
print('uzoraka u MEXICAN klasi ima: ', sum(y=='mexican'))
print('uzoraka u ITALY klasi ima: ', sum(y=='italian'))
print('uzoraka u JAPAN klasi ima: ', sum(y=='japanese'))
print('uzoraka u CHINA klasi ima: ', sum(y=='chinese'))
print('uzoraka u THAILAND klasi ima: ', sum(y=='thai'))
print('uzoraka u BRITAIN klasi ima: ', sum(y=='british'))

#moracemo pripaziti zbog nejednakog broja uzoraka kod ovih klasa ako bi smo imali neke evantualne podele skupova



#%%
drzave
y[y=='southern_us'] = 0
y[y=='french'] = 1
y[y=='greek'] = 2
y[y=='mexican'] = 3
y[y=='italian'] = 4
y[y=='japanese'] = 5
y[y=='chinese'] = 6
y[y=='thai'] = 7
y[y=='british'] = 8
y=y.astype('int')

    
#%%

def TN_calculation(conf_mat, position): 
    s = 0; 
    for i in range(0, len(conf_mat)): #<-- len(matrix) uzima visinu matrice 
        for j in range(0, len(conf_mat[0])): #<-- len(matrix[0]) uzima sirinu matrice 
            if i != position and j != position: 
                s = s + conf_mat[i, j] 
    return s

def FN_calculation(conf_mat, position):
    country_FN = 0
    for col in list(range(9)):
        country_FN = country_FN + conf_mat[position, col] 
    country_FN = country_FN - conf_mat[position, position]
    return country_FN

def FP_calculation(conf_mat, position):
    country_FP = 0
    for row in list(range(9)):
        country_FP = country_FP + conf_mat[row, position] 
    country_FP = country_FP - conf_mat[position, position]
    return country_FP

def evaluation_classif(conf_mat):
    country_names = ['british', 'chinese', 'french', 'greek', 'italian', 'japanese', 'mexican', 'southern_us', 'thai']
    for i in list(range(len(conf_mat[0]))):
        print(country_names[i] + '_TP: ', conf_mat[i, i])    
        print(country_names[i] + '_TN: ', TN_calculation(conf_mat, i))
        print(country_names[i] + '_FN: ', FN_calculation(conf_mat, i))
        print(country_names[i] + '_FP: ', FP_calculation(conf_mat, i))
        
        total = 0; 
        total = conf_mat[i, i]+TN_calculation(conf_mat, i)+FP_calculation(conf_mat, i)+FN_calculation(conf_mat, i)
        print(country_names[i] + ' precision: ', (conf_mat[i, i]) / (conf_mat[i, i] + FP_calculation(conf_mat, i)))
        print(country_names[i] + ' accuracy: ', (conf_mat[i, i] + TN_calculation(conf_mat, i)) / total)
        print(country_names[i] + ' sensitivity: ', (conf_mat[i, i]) / (conf_mat[i, i] + FN_calculation(conf_mat, i)))
        print(country_names[i] + ' specificity: ', (TN_calculation(conf_mat, i)) / (FP_calculation(conf_mat, i) + TN_calculation(conf_mat, i)))
        
        precision = (conf_mat[i, i]) / (conf_mat[i, i] + FP_calculation(conf_mat, i))
        sensitivity = (conf_mat[i, i]) / (conf_mat[i, i] + FN_calculation(conf_mat, i))
        print(country_names[i] + ' F score: ', (2*precision*sensitivity) / (precision + sensitivity))
        
        


#%%  



#k_range = range(1, 20)




# knn= 16

#k_scores5 = []
#k_scores10 = []

#for k in k_range:
#  knn = KNeighborsClassifier(n_neighbors= k)
#  scores5 = cross_val_score(knn, X_train, y_train , cv = 5, scoring= 'accuracy')
#  scores10 = cross_val_score(knn, X, y , cv = 10, scoring= 'accuracy')
#  k_scores5.append(scores5.mean())
#  k_scores10.append(scores10.mean())

#fig, axs = plt.subplots(ncols=2)
#sns.lineplot(x = k_range, y = k_scores5,ax=axs[0]);
#sns.lineplot(x = k_range, y = k_scores10,ax=axs[1]);

#max(k_scores5)
#max(k_scores10)




#%% ----------------------------- KNN KLASIFIKACIJA SA TRAŽENJEM NAJBOLJE VREDNOSTI ZA K I ODABIROM METRIKE JACCARD, DICE----------------
#  ------------------------------------------------ NAD TRENING SKUPOM -----------------------------


labels_y = y.unique()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=10)


kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc = []
#for k in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
k=16
for m in ['jaccard', 'dice']:
        indexes = kf.split(X_train, y_train)
        acc_tmp = []
        fin_conf_mat = np.zeros((len(np.unique(y_train)),len(np.unique(y_train))))
        for train_index, test_index in indexes:
            classifier = KNeighborsClassifier(n_neighbors=k, metric=m)
            classifier.fit(X_train.iloc[train_index,:], y_train.iloc[train_index])
            y_pred = classifier.predict(X_train.iloc[test_index,:])  
            acc_tmp.append(accuracy_score(y_train.iloc[test_index], y_pred))
            fin_conf_mat += confusion_matrix(y_train.iloc[test_index], y_pred, labels=labels_y)
        print('za parametre k=', k, ' i m=', m, ' tacnost je: ', np.mean(acc_tmp), ' a mat. konf. je:')
        #print(fin_conf_mat)

        disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
        disp.plot(cmap="Blues", values_format='', xticks_rotation=90)  
        plt.show()
        
        acc.append(np.mean(acc_tmp))
        
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))
evaluation_classif(fin_conf_mat.astype('int'))


#%%  --------------------------- KNN KLASIFIKACIJA UNAKRSNOM VALITACIJOM NAD TRENING SKUPOM-------------------------------------------------

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) 

indexes = kf.split(X_train, y_train) 
fin_conf_mat = np.zeros((len(np.unique(y_train)),len(np.unique(y_train))))


for train_index, test_index in indexes: 
    X1_train = X_train.iloc[train_index,:]
    X1_test = X_train.iloc[test_index,:]
    Y1_train = y_train.iloc[train_index]
    Y1_test = y_train.iloc[test_index]
    
#     scaler = StandardScaler()
#     scaler.fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=16, metric='jaccard')
    classifier.fit(X1_train, Y1_train) 
    y1_pred = classifier.predict(X1_test) # predikcije na test skupu
    conf_mat = confusion_matrix(Y1_test, y1_pred, labels=classifier.classes_) 
                                                                             
    
    disp = ConfusionMatrixDisplay.from_predictions(y_true=Y1_test, y_pred=y1_pred, labels=classifier.classes_, cmap=plt.cm.Blues, xticks_rotation=90)   
    disp.plot(cmap="Blues", values_format='.5g', xticks_rotation=90)
    plt.show()
    print("Accuracy = {}".format(accuracy_score(Y1_test, y1_pred)))
    fin_conf_mat += conf_mat
    
    

disp = ConfusionMatrixDisplay(confusion_matrix=fin_conf_mat, display_labels=classifier.classes_) 
disp.plot(cmap="Blues", values_format='.5g', xticks_rotation=90)
plt.show()
print('AVERAGE ACCURACY: ', np.trace(fin_conf_mat)/sum(sum(fin_conf_mat)))
print(' ')
evaluation_classif(fin_conf_mat.astype('int'))




#%%  ----------------------------------- KNN KLASIFIKACIJA NAD TEST SKUPOM --------------------------------------------

classifier = KNeighborsClassifier(n_neighbors=16, metric='jaccard')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred, labels=labels_y)
#print(conf_mat)

disp = ConfusionMatrixDisplay(confusion_matrix =conf_mat,  display_labels=classifier.classes_)
disp.plot(cmap="Blues", values_format='', xticks_rotation=90)  
plt.show()

print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision micro: ', precision_score(y_test, y_pred, average='micro'))
print('Precision macro: ', precision_score(y_test, y_pred, average='macro'))
print('Recall micro: ', recall_score(y_test, y_pred, average='micro'))
print('Recall macro: ', recall_score(y_test, y_pred, average='macro'))
print('F score micro: ', f1_score(y_test, y_pred, average='micro'))
print('F score macro: ', f1_score(y_test, y_pred, average='macro'))
print(labels_y)
evaluation_classif(conf_mat.astype('int'))


#%% ------------------------------------------ LOGISTICKA REGRESIJA   ------------------------------------------------------------

X_trainL, X_test, y_trainL, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=10)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc = []
recall_micro =[]
recall_macro =[]

# =============================================================================

for k in [500,1000]:
    for m in ['newton-cg', 'lbfgs', 'liblinear']:
        indexes = kf.split(X_trainL, y_trainL)
        acc_tmp = []
        recall_micro_tmp = []
        recall_macro_tmp = []
        fin_conf_mat = np.zeros((len(np.unique(y_trainL)),len(np.unique(y_trainL))))
        for train_index, test_index in indexes:
            classifier = LogisticRegression(max_iter=k, solver=m)
            classifier.fit(X_trainL.iloc[train_index,:], y_trainL.iloc[train_index])
            y_pred = classifier.predict(X_trainL.iloc[test_index,:])
            acc_tmp.append(accuracy_score(y_trainL.iloc[test_index], y_pred))
            recall_micro_tmp.append(recall_score(y_trainL.iloc[test_index], y_pred,average='micro'))
            recall_macro_tmp.append(recall_score(y_trainL.iloc[test_index], y_pred,average='macro'))
            fin_conf_mat += confusion_matrix(y_trainL.iloc[test_index], y_pred, labels=labels_y)
        print('za parametre max_iter=', k, ' i solver=', m, ' tacnost je: ', np.mean(acc_tmp), 'osetljivost micro je:',np.mean(recall_micro_tmp), 'osetljivost macro je:',np.mean(recall_macro_tmp), ' a mat. konf. je:')
        #print(fin_conf_mat)

        disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
        disp.plot(cmap="Blues", values_format='', xticks_rotation=90)  
        plt.show()
        
        acc.append(np.mean(acc_tmp))
        recall_micro.append(np.mean(recall_micro_tmp))
        recall_macro.append(np.mean(recall_macro_tmp))

print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc),', osetljivost micro u iteraciji:',np.argmax(recall_micro),', osetljivost macro u iteraciji:',np.argmax(recall_macro))
evaluation_classif(conf_mat.astype('int'))


#%%  ------------------------------------ LOGISTICKA REGRESIJA UNAKRSNOM VALIDACIJOM ---------------------------------------------------


kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) 
indexes = kf.split(X_trainL, y_trainL) 
fin_conf_mat = np.zeros((len(np.unique(y_trainL)),len(np.unique(y_trainL))))


for train_index, test_index in indexes: 
    X1_train = X_trainL.iloc[train_index,:]
    X1_test = X_trainL.iloc[test_index,:]
    Y1_train = y_trainL.iloc[train_index]
    Y1_test = y_trainL.iloc[test_index]
    
#     scaler = StandardScaler()
#     scaler.fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)

    classifier = LogisticRegression(max_iter=500, solver='newton-cg')
    classifier.fit(X1_train, Y1_train) 
    y1_pred = classifier.predict(X1_test) # predikcije na test skupu
    conf_mat = confusion_matrix(Y1_test, y1_pred, labels=classifier.classes_) 
                                                                             
    
    disp = ConfusionMatrixDisplay.from_predictions(y_true=Y1_test, y_pred=y1_pred, labels=classifier.classes_, cmap=plt.cm.Blues, xticks_rotation=90)   
    disp.plot(cmap="Blues", values_format='.5g', xticks_rotation=90)
    plt.show()
    print("Accuracy = {}".format(accuracy_score(Y1_test, y1_pred)))
    fin_conf_mat += conf_mat



disp = ConfusionMatrixDisplay(confusion_matrix=fin_conf_mat, display_labels=classifier.classes_) 
disp.plot(cmap="Blues", values_format='.5g', xticks_rotation=90)
plt.show()
print('AVERAGE ACCURACY: ', np.trace(fin_conf_mat)/sum(sum(fin_conf_mat)))
print(' ')
evaluation_classif(fin_conf_mat.astype('int'))








#%%  ---------------------------------- LOGISTICKA REGRESIJA NAD TEST SKUPOM -------------------------------------------

logreg = LogisticRegression(max_iter=500, solver='newton-cg')

logreg.fit(X_trainL,y_trainL)

y_pred=logreg.predict(X_test)


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


class_names=[0,1,2,3,4,5,6,7,8] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision micro:",metrics.precision_score(y_test, y_pred,average='micro'))
print("Precision macro:",metrics.precision_score(y_test, y_pred,average='macro'))
print("Recall micro:",metrics.recall_score(y_test, y_pred,average='micro'))
print("Recall macro:",metrics.recall_score(y_test, y_pred,average='macro'))
print("F score micro:",metrics.f1_score(y_test, y_pred,average='micro'))
print("F score macro:",metrics.f1_score(y_test, y_pred,average='macro'))
evaluation_classif(cnf_matrix.astype('int'))





