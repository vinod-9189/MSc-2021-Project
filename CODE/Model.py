import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

BBB = pd.read_csv("../datasets/FINAL.csv")
PP = pd.read_csv("../datasets/PP.csv")
NPP = pd.read_csv("../datasets/NPP.csv")

BBB.replace('Bangalore', 'Bengaluru', inplace=True)
BBB.batting_team.replace({'Rising Pune Supergiants': 'Rising Pune Supergiant'}, regex=True, inplace=True)
BBB.batting_team.replace({'Delhi Daredevils': 'Delhi Capitals'}, regex=True, inplace=True)
BBB.bowling_team.replace({'Delhi Daredevils': 'Delhi Capitals'}, regex=True, inplace=True)
BBB.bowling_team.replace({'Rising Pune Supergiants': 'Rising Pune Supergiant'}, regex=True, inplace=True)

BATSMAN = pd.read_csv("../datasets/BATSMAN.csv")
BOWLER = pd.read_csv("../datasets/BOWLER.csv")
PLAYERS = pd.read_csv("../datasets/PLAYERS.csv")

BBB = BBB.iloc[:,2:]
PP = PP.iloc[:,1:]
NPP = NPP.iloc[:,1:]

BBB = BBB[BBB["is_wicket"]==0]

#BBB = BBB[["batting_team","bowling_team"]].dropna(axis = 0)
BBB = BBB[(BBB["batting_team"].notna()) & (BBB["bowling_team"].notna())]
#print(BBB.columns)

batsman = "MS Dhoni"
bowler = "SL Malinga"

'''
'id', 'inning', 'over', 'ball', 'batsman', 'non_striker', 'bowler',
'batsman_runs', 'extra_runs', 'total_runs', 'non_boundary', 'is_wicket',
'dismissal_kind', 'player_dismissed', 'fielder', 'extras_type',
'batting_team', 'bowling_team', 'Runs In Last 6 balls',
'Fall Of Wickets In Last 6 balls', 'No_Of_Dots_In_Last_Over',
'No_Of_Singles_In_Last_Over', 'No_Of_Doubles_In_Last_Over',
'No_Of_Fours_In_Last_Over', 'No_Of_Sixes_In_Last_Over', 'Team_Score','Strike_Rate','Economy_Rate','Pressure'
'''

BBB_TEMP = BBB.iloc[:,[4,6,1,2,3,7,8,11,18,19,20,21,22,23,24,25,26,27,28]]
#NPP_TEMP = NPP.iloc[:,[4,6,1,2,3,7,8,11,18,19,20,21,22,23,24,25,26,27]]
#PP_TEMP = PP.iloc[:,[4,6,1,2,3,7,8,11,18,19,20,21,22,23,24,25,26,27]]  #batsman, bowler, over, ball, batsmanruns, extra runs, is_wicket

BATSMAN_STATS = BATSMAN.iloc[:,[0,6,7,1,2,9,10,11]]
BOWLER_STATS = BOWLER.iloc[:,[0,5,7,8,9,10]]

BATSMAN_STATS = BATSMAN_STATS[BATSMAN_STATS["batsman"] == batsman]
BOWLER_STATS = BOWLER_STATS[BOWLER_STATS["bowler"] == bowler]

STYLE = PLAYERS[PLAYERS["Player_Name"]==bowler].iloc[0]["STYLE"]
#print("\n\n",BBB_TEMP.columns)

BATSMAN_TEMP = BBB_TEMP[BBB_TEMP["batsman"]==batsman]
BOWLERS = list(PLAYERS[PLAYERS["STYLE"]==STYLE]["Player_Name"])

DATA = pd.DataFrame()

for i in range(len(BOWLERS)):
    DATA = DATA.append(BATSMAN_TEMP[BATSMAN_TEMP["bowler"]==BOWLERS[i]],ignore_index=True)

print("No of Balls "+batsman+" Faced : ",BATSMAN_TEMP["batsman"].count())
print("No of Bowlers "+batsman+" faced who were "+STYLE+" : ",len(BOWLERS))
print("No of Balls Virat Faced which were "+STYLE+" : ",DATA["bowler"].count())
print("No of Outs in "+STYLE+" : ",DATA["is_wicket"].sum())
print("Runs scored : ",DATA["batsman_runs"].sum())

#DATA = BATSMAN_TEMP
# Function for mentioning if the over belongs to powerplay
def func(row):
    if row['over'] <6 or row['over'] >15:
        return 0
    else:
        return 1

# Function to mention the no of powerplay balls in the last 6 balls
def func2(row):
    if row['over'] < 6:
        return int(100)
    elif row['over'] == 6:
        return ((6-(int(row["ball"])-1))*16.67)
    elif row['over'] == 16:
        return ((1-(int(row["ball"])-1))*16.67)
    else:
        return(0)


DATA['P/NP'] = DATA.apply(lambda row: func(row), axis=1)
DATA["Percentage P/NP"] = DATA.apply(lambda row: func2(row), axis=1)


No_Of_Matches = BBB["id"].unique().shape[0]
Match_Id = BBB["id"].unique()

'''
'batsman', 'bowler','inning', 'over', 'ball', 'batsman_runs',
'extra_runs', 'is_wicket', 'Runs In Last 6 balls',
'Fall Of Wickets In Last 6 balls', 'No_Of_Dots_In_Last_Over',
'No_Of_Singles_In_Last_Over', 'No_Of_Doubles_In_Last_Over',
'No_Of_Fours_In_Last_Over', 'No_Of_Sixes_In_Last_Over', 'Team_Score','Strike_Rate','Economy_Rate','P/NP','Percentage P/NP','Pressure'
'''
# MODEL

X= DATA.iloc[:,[2,3,4,8,9,10,11,12,13,14,15,16,17,18,19,20]]
#print(X[["over","P/NP"]])
y = DATA.iloc[:,5]
#print("\n")
#print(X.columns)
print("\n")
Target = list()
for i in range(len(DATA)):
    #if DATA.iloc[i,7] == 1:
    #Target.append("W")
    if DATA.iloc[i,5] >=4:
        Target.append(2)
    else:
        Target.append(1)

Target = pd.Series(Target)
#Target = pd.Series(y)
print(Target.unique(),"\n")

print("CLASSES AVAILABLE : ",Target.unique())


# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X = pd.DataFrame(X)

# Train and Test Split
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Target, test_size=0.2, random_state=0,stratify=Target)

def cm_analysis(Y_Test, y_pred, labels, name, ymap=None, figsize=(6,5)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      Y_Test:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original Y_Test, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        Y_Test = [ymap[yi] for yi in Y_Test]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(Y_Test, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.title("Confusion Matrix Of " + name)
    plt.show()

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

def stacking(name, model, train, y, test, ytest, n_fold):
    folds = StratifiedKFold(n_splits=n_fold, random_state=1, shuffle=True)
    test_pred = np.empty((test.shape[0], 1), float)
    train_pred = np.empty((0, 1), float)
    for train_indices, val_indices in folds.split(train, y.values):
        x_train, x_val = train.iloc[train_indices], train.iloc[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

        model.fit(X=x_train, y=y_train)
        train_pred = np.append(train_pred, model.predict(x_val))
        test_pred = np.append(test_pred, model.predict(test))
    y_pred = model.predict(test)
    print("Confusion Matrix \n", confusion_matrix(ytest, y_pred))
    print("Classification Report \n", classification_report(ytest, y_pred,zero_division=1))
    print('Accuracy of ' + name + ' classifier on test set: {:.4f}'.format(
        metrics.accuracy_score(ytest, y_pred)))
    labels = [1,2]

    ytest = np.array(ytest).reshape(66, 1)
    y_pred = np.array(y_pred).reshape(66, 1)
    print("AUC : ", roc_auc_score(ytest, model.predict_proba(test)[:, 1]))

    cm_analysis(ytest, y_pred, labels, name)
    return test_pred.reshape(-1, 1), train_pred

from sklearn.tree import DecisionTreeClassifier
models = [('Logistic Regression', LogisticRegression(solver='liblinear', random_state=7, class_weight='balanced')),
          ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=7)),
          ('SVM', SVC(gamma='auto', random_state=7,probability=True)),
          ('KNN', KNeighborsClassifier()),
          ('Decision Tree Classifier', DecisionTreeClassifier(random_state=7)),
          ('Gaussian NB', GaussianNB())
          ]
for name, model1 in models:
    test_pred1, train_pred1 = stacking(name=name, model=model1, n_fold=10, train=X_Train, test=X_Test,
                                       y=Y_Train,
                                       ytest=Y_Test)
