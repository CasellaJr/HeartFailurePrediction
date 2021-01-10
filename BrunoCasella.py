#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


heart_data = pd.read_csv('/Users/Casella/Documents/DATA SCIENCE/2 ANNO/ADVANCED MACHINE LEARNING AND KNOWLEDGE DISCOVERY/ADVANCED MACHINE LEARNING/PROGETTO FINALE/heart_failure_clinical_records_dataset.csv')
heart_data


# In[3]:


#transposition only for plotting in the report
view_for_report = heart_data.astype(object).T #i changed the datatype as object only better view
view_for_report


# In[4]:


summary_stats = heart_data.describe()
sum_stats = summary_stats.transpose() #for a better visualization
sum_stats = sum_stats.drop(['count'], axis=1) #remove the count column because it is unuseful
sum_stats


# In[5]:


#check how many males and how many females
males= sum(heart_data["sex"]==1)
males


# In[6]:


# Checking for null values
heart_data.isnull().sum()


# In[7]:


#check if there are outliers in the features

# Boxplot for age
sns.boxplot(x=heart_data.age, color = 'teal')
plt.show()

# Boxplot for creatinine_phosphokinase
sns.boxplot(x=heart_data.creatinine_phosphokinase, color = 'teal')
plt.show()

# Boxplot for ejection_fraction
sns.boxplot(x=heart_data.ejection_fraction, color = 'teal')
plt.show()

# Boxplot for platelets
sns.boxplot(x=heart_data.platelets, color = 'teal')
plt.show()

# Boxplot for serum_creatinine
sns.boxplot(x=heart_data.serum_creatinine, color = 'teal')
plt.show()

# Boxplot for serum_sodium
sns.boxplot(x=heart_data.serum_sodium, color = 'teal')
plt.show()

# Boxplot for time
sns.boxplot(x=heart_data.time, color = 'teal')
plt.show()

#No outliers in age and time. 
#However, before dealing with outliers we require knowledge about the outlier, the dataset and possibly some domain knowledge.
#Removing outliers without a good reason will not always increase accuracy. Without a deep understanding of what are the possible ranges that
#exist within each feature, removing outliers becomes tricky.


# In[8]:


#Modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


# In[9]:


#Data Vis with Plotly
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as gobj
import plotly.figure_factory as ff


# In[10]:


#Is Age and Sex an indicator for Death Event?

#age

hist_data =[heart_data["age"].values]
group_labels = ['age'] 

#fig = ff.create_distplot(hist_data, group_labels, histnorm= '', marginal="box")
#fig.update_layout(title_text='Age Distribution plot')

#fig.show()

#The same view, but using seaborn. I think Plotly produces more beautiful graphs than matplotlib and sns.
#sns.set(style='darkgrid')
#sns.displot(data=hist_data, x=heart_data["age"], kde=True, bins=55)


fig = px.histogram(hist_data, x=heart_data["age"], marginal="box", hover_data=hist_data, nbins=58,
                   title="Age distribution plot")
fig.show()


# In[11]:


fig = px.box(heart_data, x='sex', y='age', points="all", 
             title="Gender wise Age Spread - Male = 1 Female =0")

fig.show()


# In[12]:


male = heart_data[heart_data["sex"]==1]
female = heart_data[heart_data["sex"]==0]

male_survi = male[heart_data["DEATH_EVENT"]==0]
male_not = male[heart_data["DEATH_EVENT"]==1]
female_survi = female[heart_data["DEATH_EVENT"]==0]
female_not = female[heart_data["DEATH_EVENT"]==1]

#hist_data=[male_survi, male_not, female_survi, female_not]
x1=len(male_survi)
x2=len(male_not)
x3=len(female_not)
x4=len(female_survi)
hist_data=[x1,x2,x3,x4]
names=['male_survi', 'male_not', 'female_survi', 'female_not']
fig = px.bar(y=hist_data, x=names, title="Analysis on Survival - Gender")

fig.show()


# In[13]:


surv = heart_data[heart_data["DEATH_EVENT"]==0]["age"]
not_surv = heart_data[heart_data["DEATH_EVENT"]==1]["age"]

fig = go.Figure()
fig.add_trace(go.Histogram(x=surv, nbinsx=58, name="surv"))
fig.add_trace(go.Histogram(x=not_surv, nbinsx=58, name="not_surv"))
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()
#essendo due grafici sovrapposti bisogna sommare i valori delle barre


# In[14]:


fig = px.box(heart_data, y="age", x="sex", color="DEATH_EVENT", points="all",
                hover_data=heart_data.columns, title="Analysis in Age and Gender on Survival Status")
fig.show()


# In[15]:


fig = px.box(heart_data, y="age", x="smoking", color="DEATH_EVENT", points="all",
                hover_data=heart_data.columns, title="Analysis in Age and Smoking on Survival Status")

fig.show()


# In[16]:


fig = px.box(heart_data, y="age", x="diabetes", color="DEATH_EVENT", points="all", 
             hover_data=heart_data.columns, title="Analysis in Age and Diabetes on Survival Status")

fig.show()


# In[17]:


fig = px.histogram(heart_data, x="creatinine_phosphokinase", color="DEATH_EVENT", marginal="violin", 
                   hover_data=heart_data.columns)
fig.show()


# In[18]:


fig = px.histogram(heart_data, x="ejection_fraction", color="DEATH_EVENT", marginal="violin", 
                   hover_data=heart_data.columns)
fig.show()


# In[19]:


fig = px.histogram(heart_data, x="platelets", color="DEATH_EVENT", marginal="violin", 
                   hover_data=heart_data.columns)
fig.show()


# In[20]:


fig = px.histogram(heart_data, x="serum_creatinine", color="DEATH_EVENT", marginal="violin", 
                   hover_data=heart_data.columns)
fig.show()


# In[21]:


fig = px.histogram(heart_data, x="serum_sodium", color="DEATH_EVENT", marginal="violin",
                   hover_data=heart_data.columns)
fig.show()


# In[22]:


surv = heart_data[heart_data['DEATH_EVENT']==0]['serum_sodium']
not_surv = heart_data[heart_data['DEATH_EVENT']==1]['serum_sodium']
fig = go.Figure()
fig.add_trace(go.Histogram(x=surv, nbinsx=58, name="surv"))
fig.add_trace(go.Histogram(x=not_surv, nbinsx=58, name="not_surv"))
fig.update_layout(barmode='overlay', title_text="Analysis in Serum Sodium on Survival Status")
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()


# In[23]:


surv = heart_data[heart_data['DEATH_EVENT']==0]['serum_creatinine']
not_surv = heart_data[heart_data['DEATH_EVENT']==1]['serum_creatinine']
fig = go.Figure()
fig.add_trace(go.Histogram(x=surv, nbinsx=58, name="surv"))
fig.add_trace(go.Histogram(x=not_surv, nbinsx=58, name="not_surv"))
fig.update_layout(barmode='overlay', title_text="Analysis in Serum Creatinine on Survival Status")
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()


# In[24]:


surv = heart_data[heart_data['DEATH_EVENT']==0]['ejection_fraction']
not_surv = heart_data[heart_data['DEATH_EVENT']==1]['ejection_fraction']
fig = go.Figure()
fig.add_trace(go.Histogram(x=surv, nbinsx=58, name="surv"))
fig.add_trace(go.Histogram(x=not_surv, nbinsx=58, name="not_surv"))
fig.update_layout(barmode='overlay', title_text="Analysis in Ejaction Fraction on Survival Status")
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()


# In[25]:


diabetes_yes = heart_data[heart_data['diabetes']==1]
diabetes_no = heart_data[heart_data['diabetes']==0]
x1=len(diabetes_yes)
x2=len(diabetes_no)
hist_data=[x1,x2]
names=['diabetes_yes', 'diabetes_no']
fig = px.bar(y=hist_data, x=names, title="Analysis on Diabetes")
fig.show()


# In[26]:


fig = px.bar(heart_data, y='diabetes', x='DEATH_EVENT', title="Diabetes Death Event Ratio")
fig.show()


# In[27]:


diabetes_yes_survi = diabetes_yes[heart_data["DEATH_EVENT"]==0]
diabetes_yes_not_survi = diabetes_yes[heart_data["DEATH_EVENT"]==1]
diabetes_no_survi = diabetes_no[heart_data["DEATH_EVENT"]==0]
diabetes_no_not_survi = diabetes_no[heart_data["DEATH_EVENT"]==1]
x1=len(diabetes_yes_survi)
x2=len(diabetes_yes_not_survi)
x3=len(diabetes_no_survi)
x4=len(diabetes_no_not_survi)
hist_data=[x1,x2,x3,x4]
names=['diabetes_yes_survi', 'diabetes_yes_not_survi', 'diabetes_no_survi', 'diabetes_no_not_survi']
fig = px.bar(y=hist_data, x=names, title="Analysis on Survival - Diabetes")
fig.show()


# In[28]:


anaemia_yes = heart_data[heart_data['anaemia']==1]
anaemia_no = heart_data[heart_data['anaemia']==0]
x1=len(anaemia_yes)
x2=len(anaemia_no)
hist_data=[x1,x2]
names=['anaemia_yes', 'anaemia_no']
fig = px.bar(y=hist_data, x=names, title="Analysis on - Anaemia")
fig.show()


# In[29]:


fig = px.bar(heart_data, y='anaemia', x='DEATH_EVENT', title="Anaemia Death Event Ratio")
fig.show()


# In[30]:


anaemia_yes_survi = anaemia_yes[heart_data["DEATH_EVENT"]==0]
anaemia_yes_not_survi = anaemia_yes[heart_data["DEATH_EVENT"]==1]
anaemia_no_survi = anaemia_no[heart_data["DEATH_EVENT"]==0]
anaemia_no_not_survi = anaemia_no[heart_data["DEATH_EVENT"]==1]
x1=len(anaemia_yes_survi)
x2=len(anaemia_yes_not_survi)
x3=len(anaemia_no_survi)
x4=len(anaemia_no_not_survi)
hist_data=[x1,x2,x3,x4]
names=['anaemia_yes_survi', 'anaemia_yes_not_survi', 'anaemia_no_survi', 'anaemia_no_not_survi']
fig = px.bar(y=hist_data, x=names, title="Analysis on Survival - Anaemia")
fig.show()


# In[31]:


hbp_yes = heart_data[heart_data['high_blood_pressure']==1]
hbp_no = heart_data[heart_data['high_blood_pressure']==0]
x1=len(hbp_yes)
x2=len(hbp_no)
hist_data=[x1,x2]
names=['hbp_yes', 'hbp_no']
fig = px.bar(y=hist_data, x=names, title="Analysis on - High Blood Pressure")
fig.show()


# In[32]:


fig = px.bar(heart_data, y='high_blood_pressure', x='DEATH_EVENT', title="High Blood Pressure Death Event Ratio")
fig.show()


# In[33]:


hbp_yes_survi = hbp_yes[heart_data["DEATH_EVENT"]==0]
hbp_yes_not_survi = hbp_yes[heart_data["DEATH_EVENT"]==1]
hbp_no_survi = hbp_no[heart_data["DEATH_EVENT"]==0]
hbp_no_not_survi = hbp_no[heart_data["DEATH_EVENT"]==1]
x1=len(hbp_yes_survi)
x2=len(hbp_yes_not_survi)
x3=len(hbp_no_survi)
x4=len(hbp_no_not_survi)
hist_data=[x1,x2,x3,x4]
names=['hbp_yes_survi', 'hbp_yes_not_survi', 'hbp_no_survi', 'hbp_no_not_survi']
fig = px.bar(y=hist_data, x=names, title="Analysis on Survival - HBP(high blood pressure)")
fig.show()


# In[34]:


#PEARSON CORRELATION
#poi prova anche kendall tau e spearman
plt.figure(figsize=(10,10))
sns.heatmap(heart_data.corr(), vmin=-1, cmap='coolwarm', annot=True); #Pearson correlation by default

plt.figure(figsize=(10,10))
sns.heatmap(heart_data.corr(method="spearman"), vmin=-1, cmap='coolwarm', annot=True); #watch the Spearman correlation

plt.figure(figsize=(10,10))
sns.heatmap(heart_data.corr(method="kendall"), vmin=-1, cmap='coolwarm', annot=True); #watch the Kendall correlation


# In[35]:


serumdata = heart_data["serum_creatinine"]
ejecdata = heart_data["ejection_fraction"]

plt.scatter(serumdata, ejecdata, s=50, c=heart_data["DEATH_EVENT"], cmap="RdYlBu")
plt.suptitle('Scatterplot of serum creatinine versus ejection fraction', fontsize = 22)
plt.title("Blue = Dead, Red = Survived", fontsize = 14)
plt.xlabel('Serum Creatinine', fontsize = 16)
plt.ylabel('Ejection Fraction', fontsize = 16)
plt.xlim(0, 10)
plt.ylim(0, 90)
plt.show()


# In[36]:


#DATA MODELING

#ONLY 2 FEATURES (THOSE WITH HIGHEST CORRELATION), TEST SIZE 20% AND RANDOM_STATE = 1

Features = ['ejection_fraction','serum_creatinine'] #the highest correlated features with death_event
x = heart_data[Features]
y = heart_data["DEATH_EVENT"]
test_size=.20 #80% to train dataset, and 20% to test set
set_seed= 1 #in order to allor the public to replicate the same results
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=test_size, random_state=set_seed)


# In[37]:


accuracy_list = []
# logistic regression

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
log_reg_pred = log_reg.predict(x_test)
log_reg_acc = log_reg.score(x_test, y_test)
accuracy_list.append(100*log_reg_acc)


# In[38]:


print("Accuracy of Logistic Regression is : ", "{:.2f}%".format(100* log_reg_acc))


# In[39]:


cm = confusion_matrix(y_test, log_reg_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Logistic Regression Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[40]:


# svc

sv_clf = SVC()
sv_clf.fit(x_train, y_train)
sv_clf_pred = sv_clf.predict(x_test)
sv_clf_acc = sv_clf.score(x_test, y_test)
accuracy_list.append(100* sv_clf_acc)


# In[41]:


print("Accuracy of SVC is : ", "{:.2f}%".format(100* sv_clf_acc))


# In[42]:


cm = confusion_matrix(y_test, sv_clf_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("SVC Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[43]:


#Searching for the optimal number of neighbours for k-Nearest
neighbors = np.arange(1, 25) #attempts from 1 neighbour to a max of 25 neighbours
train_accuracy, test_accuracy = list(), list()

#a for cycle that for all the 1 to 25 values of neighbors, instantiate the model, then fit the model and calculate the scores
for iterator, kterator in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=kterator)
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))

plt.figure(figsize=[13, 8]) #figsize is a tuple of the width and height of the figure in inches
plt.plot(neighbors, test_accuracy, label="Testing Accuracy")
plt.plot(neighbors, train_accuracy, label="Training Accuracy")
plt.legend()
plt.title("Value VS Accuracy")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neighbors)
plt.savefig("knn_accuracies.png")
plt.show()

print("Best Accuracy is {} with K={}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))


# In[44]:


# K Neighbors Classifier with 12 neighbors 

kn12_clf = KNeighborsClassifier(n_neighbors=12)
kn12_clf.fit(x_train, y_train)
kn12_pred = kn12_clf.predict(x_test)
kn12_acc = kn12_clf.score(x_test, y_test)
accuracy_list.append(100*kn12_acc)


# In[45]:


print("Accuracy of K Neighbors Classifier is : ", "{:.2f}%".format(100* kn12_acc))


# In[46]:


cm = confusion_matrix(y_test, kn12_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("K Neighbors Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[47]:


# Decision Tree Classifier

dt_clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy')
dt_clf.fit(x_train, y_train)
dt_pred = dt_clf.predict(x_test)
dt_acc = dt_clf.score(x_test, y_test)
accuracy_list.append(100*dt_acc)


# In[48]:


print("Accuracy of Decision Tree Classifier is : ", "{:.2f}%".format(100* dt_acc))


# In[49]:


cm = confusion_matrix(y_test, dt_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Decision Tree Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[50]:


# RandomForestClassifier

r_clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)
r_clf.fit(x_train, y_train)
r_pred = r_clf.predict(x_test)
r_acc = r_clf.score(x_test, y_test)
accuracy_list.append(100*r_acc)


# In[51]:


print("Accuracy of Random Forest Classifier is : ", "{:.2f}%".format(100* r_acc))


# In[52]:


cm = confusion_matrix(y_test, r_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Random Forest Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[53]:


# ExtraTreeClassifier

ex_clf = ExtraTreesClassifier(max_features=0.5, max_depth=15, random_state=1)
ex_clf.fit(x_train, y_train)
ex_pred = ex_clf.predict(x_test)
ex_acc = ex_clf.score(x_test, y_test)
accuracy_list.append(100*ex_acc)


# In[54]:


print("Accuracy of Extra Trees Classifier is : ", "{:.2f}%".format(100* ex_acc))


# In[55]:


cm = confusion_matrix(y_test, ex_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("ExtraTrees Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[56]:


# GaussianNB

gnb_clf = GaussianNB()
gnb_clf.fit(x_train, y_train)
gnb_pred = gnb_clf.predict(x_test)
gnb_acc = gnb_clf.score(x_test, y_test)
accuracy_list.append(100*gnb_acc)


# In[57]:


print("Accuracy of GaussianNB Classifier is : ", "{:.2f}%".format(100* gnb_acc))


# In[58]:


cm = confusion_matrix(y_test, gnb_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("GaussianNB - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[59]:


# BernoulliNB

bnb_clf = BernoulliNB()
bnb_clf.fit(x_train, y_train)
bnb_pred = bnb_clf.predict(x_test)
bnb_acc = bnb_clf.score(x_test, y_test)
accuracy_list.append(100*bnb_acc)


# In[60]:


print("Accuracy of BernoulliNB Classifier is : ", "{:.2f}%".format(100* bnb_acc))


# In[61]:


cm = confusion_matrix(y_test, bnb_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("BernoulliNB - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[62]:


model_list= ["Logistic Regression", "SVC", "KNearestNeighbours12", "DecisionTree", "RandomForest", "ExtraTrees", "GaussianNB", "BernoulliNB"]


# In[63]:


plt.rcParams['figure.figsize']=20,8
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=accuracy_list, palette = "husl", saturation =2.0)
plt.xlabel('Classifier Models', fontsize = 20 )
plt.ylabel('% of Accuracy', fontsize = 20)
plt.title('Accuracy of different Classifier Models', fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


# In[64]:


#DATA MODELING

#ALL THE FEATURES (NO TIME) AND TEST SIZE 20%

x = heart_data.iloc[:, :11] #all the features except time
y = heart_data["DEATH_EVENT"]
test_size=.20 #80% to train dataset, and 20% to test set
set_seed= 1 #in order to allor the public to replicate the same results
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=test_size, random_state=set_seed)

accuracy_list = []
# logistic regression

log_reg = LogisticRegression(max_iter=140)
log_reg.fit(x_train, y_train)
log_reg_pred = log_reg.predict(x_test)
log_reg_acc = log_reg.score(x_test, y_test)
accuracy_list.append(100*log_reg_acc)

print("Accuracy of Logistic Regression is : ", "{:.2f}%".format(100* log_reg_acc))

cm = confusion_matrix(y_test, log_reg_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Logistic Regression Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# svc

sv_clf = SVC()
sv_clf.fit(x_train, y_train)
sv_clf_pred = sv_clf.predict(x_test)
sv_clf_acc = sv_clf.score(x_test, y_test)
accuracy_list.append(100* sv_clf_acc)

print("Accuracy of SVC is : ", "{:.2f}%".format(100* sv_clf_acc))

cm = confusion_matrix(y_test, sv_clf_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("SVC Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


#Searching for the optimal number of neighbours for k-Nearest
neighbors = np.arange(1, 25) #attempts from 1 neighbour to a max of 25 neighbours
train_accuracy, test_accuracy = list(), list()

#a for cycle that for all the 1 to 25 values of neighbors, instantiate the model, then fit the model and calculate the scores
for iterator, kterator in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=kterator)
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))

plt.figure(figsize=[13, 8]) #figsize is a tuple of the width and height of the figure in inches
plt.plot(neighbors, test_accuracy, label="Testing Accuracy")
plt.plot(neighbors, train_accuracy, label="Training Accuracy")
plt.legend()
plt.title("Value VS Accuracy")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neighbors)
plt.savefig("knn_accuracies.png")
plt.show()

print("Best Accuracy is {} with K={}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))

# K Neighbors Classifier with 18 neighbors 

kn18_clf = KNeighborsClassifier(n_neighbors=18)
kn18_clf.fit(x_train, y_train)
kn18_pred = kn18_clf.predict(x_test)
kn18_acc = kn18_clf.score(x_test, y_test)
accuracy_list.append(100*kn18_acc)

print("Accuracy of K Neighbors Classifier is : ", "{:.2f}%".format(100* kn18_acc))

cm = confusion_matrix(y_test, kn18_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("K Neighbors Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# Decision Tree Classifier

dt_clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy')
dt_clf.fit(x_train, y_train)
dt_pred = dt_clf.predict(x_test)
dt_acc = dt_clf.score(x_test, y_test)
accuracy_list.append(100*dt_acc)

print("Accuracy of Decision Tree Classifier is : ", "{:.2f}%".format(100* dt_acc))

cm = confusion_matrix(y_test, dt_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Decision Tree Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# RandomForestClassifier

r_clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)
r_clf.fit(x_train, y_train)
r_pred = r_clf.predict(x_test)
r_acc = r_clf.score(x_test, y_test)
accuracy_list.append(100*r_acc)

print("Accuracy of Random Forest Classifier is : ", "{:.2f}%".format(100* r_acc))

cm = confusion_matrix(y_test, r_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Random Forest Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# ExtraTreeClassifier

ex_clf = ExtraTreesClassifier(max_features=0.5, max_depth=15, random_state=1)
ex_clf.fit(x_train, y_train)
ex_pred = ex_clf.predict(x_test)
ex_acc = ex_clf.score(x_test, y_test)
accuracy_list.append(100*ex_acc)

print("Accuracy of Extra Trees Classifier is : ", "{:.2f}%".format(100* ex_acc))

cm = confusion_matrix(y_test, ex_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("ExtraTrees Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# GaussianNB

gnb_clf = GaussianNB()
gnb_clf.fit(x_train, y_train)
gnb_pred = gnb_clf.predict(x_test)
gnb_acc = gnb_clf.score(x_test, y_test)
accuracy_list.append(100*gnb_acc)

print("Accuracy of GaussianNB Classifier is : ", "{:.2f}%".format(100* gnb_acc))

cm = confusion_matrix(y_test, gnb_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("GaussianNB - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# BernoulliNB

bnb_clf = BernoulliNB()
bnb_clf.fit(x_train, y_train)
bnb_pred = bnb_clf.predict(x_test)
bnb_acc = bnb_clf.score(x_test, y_test)
accuracy_list.append(100*bnb_acc)

print("Accuracy of BernoulliNB Classifier is : ", "{:.2f}%".format(100* bnb_acc))

cm = confusion_matrix(y_test, bnb_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("BernoulliNB - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

model_list= ["Logistic Regression", "SVC", "KNearestNeighbours18", "DecisionTree", "RandomForest", "ExtraTrees", "GaussianNB", "BernoulliNB"]

plt.rcParams['figure.figsize']=20,8
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=accuracy_list, palette = "husl", saturation =2.0)
plt.xlabel('Classifier Models', fontsize = 20 )
plt.ylabel('% of Accuracy', fontsize = 20)
plt.title('Accuracy of different Classifier Models', fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


# In[65]:


#DATA MODELING

#ONLY 2 FEATURES + TIME

Features = ['ejection_fraction', 'serum_creatinine', 'time'] #the highest correlated features with death_event
x = heart_data[Features]
y = heart_data["DEATH_EVENT"]
test_size=.20 #80% to train dataset, and 20% to test set
set_seed= 1 #in order to allor the public to replicate the same results
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=test_size, random_state=set_seed)

accuracy_list = []
# logistic regression

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
log_reg_pred = log_reg.predict(x_test)
log_reg_acc = log_reg.score(x_test, y_test)
accuracy_list.append(100*log_reg_acc)

print("Accuracy of Logistic Regression is : ", "{:.2f}%".format(100* log_reg_acc))

cm = confusion_matrix(y_test, log_reg_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Logistic Regression Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# svc

sv_clf = SVC()
sv_clf.fit(x_train, y_train)
sv_clf_pred = sv_clf.predict(x_test)
sv_clf_acc = sv_clf.score(x_test, y_test)
accuracy_list.append(100* sv_clf_acc)

print("Accuracy of SVC is : ", "{:.2f}%".format(100* sv_clf_acc))

cm = confusion_matrix(y_test, sv_clf_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("SVC Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


#Searching for the optimal number of neighbours for k-Nearest
neighbors = np.arange(1, 25) #attempts from 1 neighbour to a max of 25 neighbours
train_accuracy, test_accuracy = list(), list()

#a for cycle that for all the 1 to 25 values of neighbors, instantiate the model, then fit the model and calculate the scores
for iterator, kterator in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=kterator)
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))

plt.figure(figsize=[13, 8]) #figsize is a tuple of the width and height of the figure in inches
plt.plot(neighbors, test_accuracy, label="Testing Accuracy")
plt.plot(neighbors, train_accuracy, label="Training Accuracy")
plt.legend()
plt.title("Value VS Accuracy")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neighbors)
plt.savefig("knn_accuracies.png")
plt.show()

print("Best Accuracy is {} with K={}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))

# K Neighbors Classifier with 6 neighbors 

kn6_clf = KNeighborsClassifier(n_neighbors=6)
kn6_clf.fit(x_train, y_train)
kn6_pred = kn6_clf.predict(x_test)
kn6_acc = kn6_clf.score(x_test, y_test)
accuracy_list.append(100*kn6_acc)

print("Accuracy of K Neighbors Classifier is : ", "{:.2f}%".format(100* kn6_acc))

cm = confusion_matrix(y_test, kn6_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("K Neighbors Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# Decision Tree Classifier

dt_clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy')
dt_clf.fit(x_train, y_train)
dt_pred = dt_clf.predict(x_test)
dt_acc = dt_clf.score(x_test, y_test)
accuracy_list.append(100*dt_acc)

print("Accuracy of Decision Tree Classifier is : ", "{:.2f}%".format(100* dt_acc))

cm = confusion_matrix(y_test, dt_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Decision Tree Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# RandomForestClassifier

r_clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)
r_clf.fit(x_train, y_train)
r_pred = r_clf.predict(x_test)
r_acc = r_clf.score(x_test, y_test)
accuracy_list.append(100*r_acc)

print("Accuracy of Random Forest Classifier is : ", "{:.2f}%".format(100* r_acc))

cm = confusion_matrix(y_test, r_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Random Forest Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# ExtraTreeClassifier

ex_clf = ExtraTreesClassifier(max_features=0.5, max_depth=15, random_state=1)
ex_clf.fit(x_train, y_train)
ex_pred = ex_clf.predict(x_test)
ex_acc = ex_clf.score(x_test, y_test)
accuracy_list.append(100*ex_acc)

print("Accuracy of Extra Trees Classifier is : ", "{:.2f}%".format(100* ex_acc))

cm = confusion_matrix(y_test, ex_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("ExtraTrees Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# GaussianNB

gnb_clf = GaussianNB()
gnb_clf.fit(x_train, y_train)
gnb_pred = gnb_clf.predict(x_test)
gnb_acc = gnb_clf.score(x_test, y_test)
accuracy_list.append(100*gnb_acc)

print("Accuracy of GaussianNB Classifier is : ", "{:.2f}%".format(100* gnb_acc))

cm = confusion_matrix(y_test, gnb_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("GaussianNB - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# BernoulliNB

bnb_clf = BernoulliNB()
bnb_clf.fit(x_train, y_train)
bnb_pred = bnb_clf.predict(x_test)
bnb_acc = bnb_clf.score(x_test, y_test)
accuracy_list.append(100*bnb_acc)

print("Accuracy of BernoulliNB Classifier is : ", "{:.2f}%".format(100* bnb_acc))

cm = confusion_matrix(y_test, bnb_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("BernoulliNB - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

model_list= ["Logistic Regression", "SVC", "KNearestNeighbours6", "DecisionTree", "RandomForest", "ExtraTrees", "GaussianNB", "BernoulliNB"]

plt.rcParams['figure.figsize']=20,8
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=accuracy_list, palette = "husl", saturation =2.0)
plt.xlabel('Classifier Models', fontsize = 20 )
plt.ylabel('% of Accuracy', fontsize = 20)
plt.title('Accuracy of different Classifier Models', fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


# In[66]:


#DATA MODELING

#ALL THE FEATURES + TIME 

x = heart_data.iloc[:, :12] #all the features + time
y = heart_data["DEATH_EVENT"]
test_size=.20 #80% to train dataset, and 20% to test set
set_seed= 1 #in order to allor the public to replicate the same results
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=test_size, random_state=set_seed)

accuracy_list = []
# logistic regression

log_reg = LogisticRegression(max_iter=140)
log_reg.fit(x_train, y_train)
log_reg_pred = log_reg.predict(x_test)
log_reg_acc = log_reg.score(x_test, y_test)
accuracy_list.append(100*log_reg_acc)

print("Accuracy of Logistic Regression is : ", "{:.2f}%".format(100* log_reg_acc))

cm = confusion_matrix(y_test, log_reg_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Logistic Regression Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# svc

sv_clf = SVC()
sv_clf.fit(x_train, y_train)
sv_clf_pred = sv_clf.predict(x_test)
sv_clf_acc = sv_clf.score(x_test, y_test)
accuracy_list.append(100* sv_clf_acc)

print("Accuracy of SVC is : ", "{:.2f}%".format(100* sv_clf_acc))

cm = confusion_matrix(y_test, sv_clf_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("SVC Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

#Searching for the optimal number of neighbours for k-nearest 
neighbors = np.arange(1, 25) #attempts from 1 neighbour to a max of 25 neighbours
train_accuracy, test_accuracy = list(), list()

#a for cycle that for all the 1 to 25 values of neighbors, instantiate the model, then fit the model and calculate the scores
for iterator, kterator in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=kterator)
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))

plt.figure(figsize=[13, 8]) #figsize is a tuple of the width and height of the figure in inches
plt.plot(neighbors, test_accuracy, label="Testing Accuracy")
plt.plot(neighbors, train_accuracy, label="Training Accuracy")
plt.legend()
plt.title("Value VS Accuracy")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neighbors)
plt.savefig("knn_accuracies.png")
plt.show()

print("Best Accuracy is {} with K={}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))

# K Neighbors Classifier with 18 neighbors 

kn18_clf = KNeighborsClassifier(n_neighbors=18)
kn18_clf.fit(x_train, y_train)
kn18_pred = kn18_clf.predict(x_test)
kn18_acc = kn18_clf.score(x_test, y_test)
accuracy_list.append(100*kn18_acc)

print("Accuracy of K Neighbors Classifier is : ", "{:.2f}%".format(100* kn18_acc))

cm = confusion_matrix(y_test, kn18_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("K Neighbors Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# Decision Tree Classifier

dt_clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy')
dt_clf.fit(x_train, y_train)
dt_pred = dt_clf.predict(x_test)
dt_acc = dt_clf.score(x_test, y_test)
accuracy_list.append(100*dt_acc)

print("Accuracy of Decision Tree Classifier is : ", "{:.2f}%".format(100* dt_acc))

cm = confusion_matrix(y_test, dt_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Decision Tree Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# RandomForestClassifier

r_clf = RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)
r_clf.fit(x_train, y_train)
r_pred = r_clf.predict(x_test)
r_acc = r_clf.score(x_test, y_test)
accuracy_list.append(100*r_acc)

print("Accuracy of Random Forest Classifier is : ", "{:.2f}%".format(100* r_acc))

cm = confusion_matrix(y_test, r_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Random Forest Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# ExtraTreeClassifier

ex_clf = ExtraTreesClassifier(max_features=0.5, max_depth=15, random_state=1)
ex_clf.fit(x_train, y_train)
ex_pred = ex_clf.predict(x_test)
ex_acc = ex_clf.score(x_test, y_test)
accuracy_list.append(100*ex_acc)

print("Accuracy of Extra Trees Classifier is : ", "{:.2f}%".format(100* ex_acc))

cm = confusion_matrix(y_test, ex_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("ExtraTrees Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# GaussianNB

gnb_clf = GaussianNB()
gnb_clf.fit(x_train, y_train)
gnb_pred = gnb_clf.predict(x_test)
gnb_acc = gnb_clf.score(x_test, y_test)
accuracy_list.append(100*gnb_acc)

print("Accuracy of GaussianNB Classifier is : ", "{:.2f}%".format(100* gnb_acc))

cm = confusion_matrix(y_test, gnb_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("GaussianNB - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

# BernoulliNB

bnb_clf = BernoulliNB()
bnb_clf.fit(x_train, y_train)
bnb_pred = bnb_clf.predict(x_test)
bnb_acc = bnb_clf.score(x_test, y_test)
accuracy_list.append(100*bnb_acc)

print("Accuracy of BernoulliNB Classifier is : ", "{:.2f}%".format(100* bnb_acc))

cm = confusion_matrix(y_test, bnb_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("BernoulliNB - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

model_list= ["Logistic Regression", "SVC", "KNearestNeighbours18", "DecisionTree", "RandomForest", "ExtraTrees", "GaussianNB", "BernoulliNB"]

plt.rcParams['figure.figsize']=20,8
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=accuracy_list, palette = "husl", saturation =2.0)
plt.xlabel('Classifier Models', fontsize = 20 )
plt.ylabel('% of Accuracy', fontsize = 20)
plt.title('Accuracy of different Classifier Models', fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


# In[67]:


#CLUSTER ANALYSIS
#SILHOUETTE COEFFICIENT
#ALL THE FEATURES EXCEPT DEATH_EVENT
Features = heart_data.drop(["DEATH_EVENT"], axis = 1).columns

X = heart_data[Features]

sc = []
for i in range(2, 25):
    kmeans = Pipeline([("scaling",StandardScaler()),("clustering",KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0))]).fit(X)
    score = silhouette_score(X, kmeans["clustering"].labels_)
    sc.append(score)
plt.plot(range(2, 25), sc, marker = "o")
plt.title('Silhouette', fontsize = 22)
plt.xlabel('Number of clusters', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.show()


# In[68]:


#FOR VISUALIZATION. I PLOT THE CASE OF 2 CLUSTERS
kmeans = Pipeline([("scaling",StandardScaler()),("clustering",KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0))]).fit(X)
plt.style.use("seaborn-whitegrid")

pca = Pipeline([("scaling",StandardScaler()),("decompositioning",PCA(n_components = 2))]).fit(X)

X2D = pca.transform(X)

plt.scatter(X2D[:,0],X2D[:,1], c = kmeans["clustering"].labels_, cmap = "RdYlBu")
plt.colorbar();


# In[69]:


#NOW I MAKE PREDICTION AND DIVIDE THE OBSERVATIONS IN 2 CLUSTERS
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
cluster_result = kmeans.predict(X)
new_array = np.concatenate((heart_data, cluster_result[:,None]), axis = 1)
comparison = pd.DataFrame(new_array)
pd.set_option("display.max_rows", None, "display.max_columns", None) #to show all the rows
final_comp = comparison.iloc[:, 12:14]
final_comp


# In[70]:


#WITH A CONFUSION MATRIX I SEE THE DIFFERENCES BETWEEN REAL VALUES AND CLUSTERS
true_values = heart_data["DEATH_EVENT"]
cluster_values = comparison.iloc[:,13]
cm = confusion_matrix(true_values, cluster_values)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Cluster - Confusion Matrix")
plt.xticks(range(2), ["Cluster 1","Cluster 2"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[71]:


#SAME STEPS BUT ONYL 2 FEATURES
Features = ["ejection_fraction", "serum_creatinine"]
X = heart_data[Features]

sc = []
for i in range(2, 25):
    kmeans = Pipeline([("scaling",StandardScaler()),("clustering",KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0))]).fit(X)
    score = silhouette_score(X, kmeans["clustering"].labels_)
    sc.append(score)
plt.plot(range(2, 25), sc, marker = "o")
plt.title('Silhouette', fontsize=22)
plt.xlabel('Number of clusters', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.show()


# In[72]:


kmeans = Pipeline([("scaling",StandardScaler()),("clustering",KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0))]).fit(X)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

pca = Pipeline([("scaling",StandardScaler()),("decompositioning",PCA(n_components = 2))]).fit(X)

X2D = pca.transform(X)

plt.scatter(X2D[:,0],X2D[:,1], c = kmeans["clustering"].labels_, cmap = "RdYlBu")
plt.colorbar();


# In[73]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
cluster_result = kmeans.predict(X)
new_array = np.concatenate((heart_data, cluster_result[:,None]), axis = 1)
comparison = pd.DataFrame(new_array)
pd.set_option("display.max_rows", None, "display.max_columns", None) #to show all the rows
final_comp = comparison.iloc[:, 12:14]
final_comp


# In[74]:


true_values = heart_data["DEATH_EVENT"]
cluster_values = comparison.iloc[:,13]
cm = confusion_matrix(true_values, cluster_values)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Cluster - Confusion Matrix")
plt.xticks(range(2), ["Cluster 1","Cluster 2"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()


# In[ ]:




