
# Identifiying Fraud in Enron Dataset

Enron was one of the biggest energy companies in 2000. By 2002, the company got bankrupted due to high level of fraud conducted by some of the employees. Following the investigation, a big dataset containing financial and email information on the employees was made publicly available. 

My goal here is to make use of this dataset in order to develop a machine learning algorithm that could identify persons of interests among the people in data set. A person of interest (POI) is someone who was indicted for fraud, settled with the government, or testified in exchange for immunity.


```python
# Import all necessary libraries
import sys
import pickle
import matplotlib.pyplot
from numpy import mean
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from pprint import pprint
from feature_format import featureFormat, targetFeatureSplit
```


```python
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
```

## Data Structure

Let's first start with analyzing the data.


```python

```


```python
# Check number of people and attributes in the dataset
print 'Number of people:', len(data_dict)

counter = 0
for i in data_dict:
    if counter < 1:
        print 'Number of attributes per person:', len(data_dict[i])
        counter +=1
```

    Number of people: 146
    Number of attributes per person: 21



```python
# Check number of POIs and Non-POIs
counter = 0
for i in data_dict:
    if data_dict[i]['poi']:
        counter +=1
print "Number of Poi's:", counter
print "Number of Non-Poi's:", len(data_dict) - counter
```

    Number of Poi's: 18
    Number of Non-Poi's: 128



```python
# Check number of NaN values
counter_nan = {}
for i in data_dict:
    for a in data_dict[i]:
        if data_dict[i][a] == 'NaN':
            if a not in counter_nan:
                counter_nan[a] = 1
            else:
                counter_nan[a] += 1 
#            counter_nan += 1
            
print "Number of NaN values:", sum(counter_nan.values())
print ''
print 'Features with most NaN values:' 
print ''
pprint(sorted(counter_nan.items(), key = lambda x:x[1], reverse = True))
```

    Number of NaN values: 1358
    
    Features with most NaN values:
    
    [('loan_advances', 142),
     ('director_fees', 129),
     ('restricted_stock_deferred', 128),
     ('deferral_payments', 107),
     ('deferred_income', 97),
     ('long_term_incentive', 80),
     ('bonus', 64),
     ('to_messages', 60),
     ('shared_receipt_with_poi', 60),
     ('from_poi_to_this_person', 60),
     ('from_messages', 60),
     ('from_this_person_to_poi', 60),
     ('other', 53),
     ('salary', 51),
     ('expenses', 51),
     ('exercised_stock_options', 44),
     ('restricted_stock', 36),
     ('email_address', 35),
     ('total_payments', 21),
     ('total_stock_value', 20)]


## Outliers / Irregularities

As next, let's investigate if there are any outliers/irregularities that we need to remove.


```python
# Visualize the data to check for outliers
features = ["salary", "bonus"]
data_outlier = featureFormat(data_dict, features)



for point in data_outlier:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

print 'Printing complete dataset:'
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
```

    Printing complete dataset:



![png](output_12_1.png)


So, we have one datapoint that is most likely a outlier. Let's investigate further:


```python
for key in data_dict:
    if data_dict[key]['bonus'] <> 'NaN' and data_dict[key]['bonus'] > 20000000:
        print 'Employee names with bonus over 20000000:', key
```

    Employee names with bonus over 20000000: TOTAL


So, our outlier is actually a row which sums up values for all people. We can remove it from the dataset.


```python
# Remove TOTAL
data_dict.pop('TOTAL', 0)  
```




    {'bonus': 97343619,
     'deferral_payments': 32083396,
     'deferred_income': -27992891,
     'director_fees': 1398517,
     'email_address': 'NaN',
     'exercised_stock_options': 311764000,
     'expenses': 5235198,
     'from_messages': 'NaN',
     'from_poi_to_this_person': 'NaN',
     'from_this_person_to_poi': 'NaN',
     'loan_advances': 83925000,
     'long_term_incentive': 48521928,
     'other': 42667589,
     'poi': False,
     'restricted_stock': 130322299,
     'restricted_stock_deferred': -7576788,
     'salary': 26704229,
     'shared_receipt_with_poi': 'NaN',
     'to_messages': 'NaN',
     'total_payments': 309886585,
     'total_stock_value': 434509511}



Let's make sure there aren't any other unusual entries in the dataset.


```python
# Check the names for irregularities
for key in data_dict:
    print key
```

    METTS MARK
    BAXTER JOHN C
    ELLIOTT STEVEN
    CORDES WILLIAM R
    HANNON KEVIN P
    MORDAUNT KRISTINA M
    MEYER ROCKFORD G
    MCMAHON JEFFREY
    HORTON STANLEY C
    PIPER GREGORY F
    HUMPHREY GENE E
    UMANOFF ADAM S
    BLACHMAN JEREMY M
    SUNDE MARTIN
    GIBBS DANA R
    LOWRY CHARLES P
    COLWELL WESLEY
    MULLER MARK S
    JACKSON CHARLENE R
    WESTFAHL RICHARD K
    WALTERS GARETH W
    WALLS JR ROBERT H
    KITCHEN LOUISE
    CHAN RONNIE
    BELFER ROBERT
    SHANKMAN JEFFREY A
    WODRASKA JOHN
    BERGSIEKER RICHARD P
    URQUHART JOHN A
    BIBI PHILIPPE A
    RIEKER PAULA H
    WHALEY DAVID A
    BECK SALLY W
    HAUG DAVID L
    ECHOLS JOHN B
    MENDELSOHN JOHN
    HICKERSON GARY J
    CLINE KENNETH W
    LEWIS RICHARD
    HAYES ROBERT E
    MCCARTY DANNY J
    KOPPER MICHAEL J
    LEFF DANIEL P
    LAVORATO JOHN J
    BERBERIAN DAVID
    DETMERING TIMOTHY J
    WAKEHAM JOHN
    POWERS WILLIAM
    GOLD JOSEPH
    BANNANTINE JAMES M
    DUNCAN JOHN H
    SHAPIRO RICHARD S
    SHERRIFF JOHN R
    SHELBY REX
    LEMAISTRE CHARLES
    DEFFNER JOSEPH M
    KISHKILL JOSEPH G
    WHALLEY LAWRENCE G
    MCCONNELL MICHAEL S
    PIRO JIM
    DELAINEY DAVID W
    SULLIVAN-SHAKLOVITZ COLLEEN
    WROBEL BRUCE
    LINDHOLM TOD A
    MEYER JEROME J
    LAY KENNETH L
    BUTTS ROBERT H
    OLSON CINDY K
    MCDONALD REBECCA
    CUMBERLAND MICHAEL S
    GAHN ROBERT S
    MCCLELLAN GEORGE
    HERMANN ROBERT J
    SCRIMSHAW MATTHEW
    GATHMANN WILLIAM D
    HAEDICKE MARK E
    BOWEN JR RAYMOND M
    GILLIS JOHN
    FITZGERALD JAY L
    MORAN MICHAEL P
    REDMOND BRIAN L
    BAZELIDES PHILIP J
    BELDEN TIMOTHY N
    DURAN WILLIAM D
    THORN TERENCE H
    FASTOW ANDREW S
    FOY JOE
    CALGER CHRISTOPHER F
    RICE KENNETH D
    KAMINSKI WINCENTY J
    LOCKHART EUGENE E
    COX DAVID
    OVERDYKE JR JERE C
    PEREIRA PAULO V. FERRAZ
    STABLER FRANK
    SKILLING JEFFREY K
    BLAKE JR. NORMAN P
    SHERRICK JEFFREY B
    PRENTICE JAMES
    GRAY RODNEY
    PICKERING MARK R
    THE TRAVEL AGENCY IN THE PARK
    NOLES JAMES L
    KEAN STEVEN J
    FOWLER PEGGY
    WASAFF GEORGE
    WHITE JR THOMAS E
    CHRISTODOULOU DIOMEDES
    ALLEN PHILLIP K
    SHARP VICTORIA T
    JAEDICKE ROBERT
    WINOKUR JR. HERBERT S
    BROWN MICHAEL
    BADUM JAMES P
    HUGHES JAMES A
    REYNOLDS LAWRENCE
    DIMICHELE RICHARD G
    BHATNAGAR SANJAY
    CARTER REBECCA C
    BUCHANAN HAROLD G
    YEAP SOON
    MURRAY JULIA H
    GARLAND C KEVIN
    DODSON KEITH
    YEAGER F SCOTT
    HIRKO JOSEPH
    DIETRICH JANET R
    DERRICK JR. JAMES V
    FREVERT MARK A
    PAI LOU L
    BAY FRANKLIN R
    HAYSLETT RODERICK J
    FUGH JOHN L
    FALLON JAMES B
    KOENIG MARK E
    SAVAGE FRANK
    IZZO LAWRENCE L
    TILNEY ELIZABETH A
    MARTIN AMANDA K
    BUY RICHARD B
    GRAMM WENDY L
    CAUSEY RICHARD A
    TAYLOR MITCHELL S
    DONAHUE JR JEFFREY M
    GLISAN JR BEN F


So, we have an entry that is called 'THE TRAVEL AGENCY IN THE PARK' among other employees. I believe it is safe to remove this entry as well.


```python
# Remove 'THE TRAVEL AGENCY IN THE PARK'
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
```




    {'bonus': 'NaN',
     'deferral_payments': 'NaN',
     'deferred_income': 'NaN',
     'director_fees': 'NaN',
     'email_address': 'NaN',
     'exercised_stock_options': 'NaN',
     'expenses': 'NaN',
     'from_messages': 'NaN',
     'from_poi_to_this_person': 'NaN',
     'from_this_person_to_poi': 'NaN',
     'loan_advances': 'NaN',
     'long_term_incentive': 'NaN',
     'other': 362096,
     'poi': False,
     'restricted_stock': 'NaN',
     'restricted_stock_deferred': 'NaN',
     'salary': 'NaN',
     'shared_receipt_with_poi': 'NaN',
     'to_messages': 'NaN',
     'total_payments': 362096,
     'total_stock_value': 'NaN'}



As last, let's check if there is any employee who only consists of NaN values.


```python
for person in data_dict:
    if all(x == 'NaN' or x is False for x in data_dict[person].values()):
        print person
```

    LOCKHART EUGENE E


Since there is no information about this particular employee, we can also remove the entry.


```python
# Remove 'LOCKHART EUGENE E'
data_dict.pop('LOCKHART EUGENE E', 0)
```




    {'bonus': 'NaN',
     'deferral_payments': 'NaN',
     'deferred_income': 'NaN',
     'director_fees': 'NaN',
     'email_address': 'NaN',
     'exercised_stock_options': 'NaN',
     'expenses': 'NaN',
     'from_messages': 'NaN',
     'from_poi_to_this_person': 'NaN',
     'from_this_person_to_poi': 'NaN',
     'loan_advances': 'NaN',
     'long_term_incentive': 'NaN',
     'other': 'NaN',
     'poi': False,
     'restricted_stock': 'NaN',
     'restricted_stock_deferred': 'NaN',
     'salary': 'NaN',
     'shared_receipt_with_poi': 'NaN',
     'to_messages': 'NaN',
     'total_payments': 'NaN',
     'total_stock_value': 'NaN'}



So now, let's take another look how the plot looks now.


```python
data_outlier = featureFormat(data_dict, features)

for point in data_outlier:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

print 'Printing dataset without outliers'
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
```

    Printing dataset without outliers



![png](output_26_1.png)


Now, it looks more reasonable.

## Features Selection

We have the following attributes in the dataset:

#### Financial 
- deferral_payments
- expenses
- long_term_incentive
- deferred_income
- restricted_stock_deferred
- loan_advances
- restricted_stock
- salary
- total_payments
- exercised_stock_options
- other
- director_fees
- bonus
- total_stock_value

#### Email
- email_address
- to_messages
- from_messages
- from_poi_to_this_person
- from_this_person_to_poi
- shared_receipt_with_poi

#### POI
- POI

Basicallly, the financial attributes give information about the financial status of the employees, whereas Email attributes captures the mailing interactions between them. Lastly, POI indicates whether a person is marked as POI, so we can use POI as our label to test performance of different machine learning algorithms.

Since the people who got involved in the fraud should have done this for monetary reasons on the first place, I believe all of the financial features might be relevant for indicating the Pois. So I will take all of them initially as my features. 


```python
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
```

On the other hand, I will create two additional features from the email features, ‘fraction_email_from_poi’ and ‘fraction_email_to_poi’ to see the level of interaction with other pois. Basically fraction_email_from_poi is supposed to show fraction of all incoming emails that come from POIs and fraction_email_to_poi is supposed to show fraction of all outgoing emails that were sent to POIs.


```python
### Create new feature(s)
for i in data_dict:
    employee = data_dict[i]
    if employee['from_poi_to_this_person'] <> 'NaN' and employee['to_messages'] <> 'NaN':
        data_dict[i]['fraction_email_from_poi'] = float(employee['from_poi_to_this_person']) / float(employee['to_messages'])
    else:
        data_dict[i]['fraction_email_from_poi'] = 0
    if employee['from_this_person_to_poi'] <> 'NaN' and employee['from_messages'] <> 'NaN':
        data_dict[i]['fraction_email_to_poi'] = float(employee['from_this_person_to_poi']) / float(employee['from_messages'])
    else:
        data_dict[i]['fraction_email_to_poi'] = 0
        
        
### add the new features to the features_list
features_list.extend(('fraction_email_from_poi','fraction_email_to_poi'))
```


```python
### Store to my_dataset for easy export below.
my_dataset = data_dict
```


```python
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print 'Starting features:' 
pprint(features_list)
```

    Starting features:
    ['poi',
     'salary',
     'deferral_payments',
     'total_payments',
     'loan_advances',
     'bonus',
     'restricted_stock_deferred',
     'deferred_income',
     'total_stock_value',
     'expenses',
     'exercised_stock_options',
     'other',
     'long_term_incentive',
     'restricted_stock',
     'director_fees',
     'fraction_email_from_poi',
     'fraction_email_to_poi']


So I will take all these features and run SelectKBest with default parameters to find the 10 strongest features for finding Pois.



```python
# Use SelectKBest to find most relevant features
best_10_features = SelectKBest()
best_10_features.fit(features,labels)

results_10_list = zip(best_10_features.get_support(), features_list[1:], best_10_features.scores_)
results_10_list = sorted(results_10_list, key=lambda x: x[2], reverse=True)
print "best 10 features:" 
pprint(results_10_list)
```

    best 10 features:
    [(True, 'exercised_stock_options', 24.815079733218194),
     (True, 'total_stock_value', 24.182898678566879),
     (True, 'bonus', 20.792252047181535),
     (True, 'salary', 18.289684043404513),
     (True, 'fraction_email_to_poi', 16.409712548035792),
     (True, 'deferred_income', 11.458476579280369),
     (True, 'long_term_incentive', 9.9221860131898225),
     (True, 'restricted_stock', 9.2128106219771002),
     (True, 'total_payments', 8.7727777300916756),
     (True, 'loan_advances', 7.1840556582887247),
     (False, 'expenses', 6.0941733106389453),
     (False, 'other', 4.1874775069953749),
     (False, 'fraction_email_from_poi', 3.1280917481567192),
     (False, 'director_fees', 2.1263278020077054),
     (False, 'deferral_payments', 0.22461127473600989),
     (False, 'restricted_stock_deferred', 0.065499652909942141)]


So one of my newly created features, **fraction_email_from_poi** does not make it to the first ten. On the other hand fraction_email_to_poi ended up being the 5th important feature. As next, I would like to create 3 alternative feature lists, one with 10 features, one with 5 and one with 3 features in order to compare the effect of number of features on te final algortihm.


```python
features_10_list = ['poi', 'exercised_stock_options', 'total_stock_value' , 'bonus', 'salary', 'fraction_email_to_poi', 'deferred_income', 'long_term_incentive', 'restricted_stock', 'total_payments', 'loan_advances']
data_10 = featureFormat(my_dataset, features_10_list, sort_keys = True)
labels_10, features_10 = targetFeatureSplit(data_10)


features_5_list = ['poi', 'exercised_stock_options', 'total_stock_value' , 'bonus', 'salary', 'fraction_email_to_poi']
data_5 = featureFormat(my_dataset, features_5_list, sort_keys = True)
labels_5, features_5 = targetFeatureSplit(data_5)



features_list_3 = ['poi', 'exercised_stock_options', 'total_stock_value' , 'bonus']
data_3 = featureFormat(my_dataset, features_list_3, sort_keys = True)
labels_3, features_3 = targetFeatureSplit(data_3)
```

Since many of the features have a huge range across different employees, I will make use of the MinMaxScaler and scale all of the features in all the lists.


```python
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features_10 = scaler.fit_transform(features_10)
features_5 = scaler.fit_transform(features_5)
features_3 = scaler.fit_transform(features_3)
```

## Classifier Selection

I would like to test the following machine learning algorithms for my dataset:

- Naive Bayes
- Kmeans
- Support Vector Machine
- Random Forest


```python
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.001)

# Tune parameters to see the effect
from sklearn.svm import SVC
s1_clf = SVC(kernel='rbf', C=1000,gamma = 0.0001,random_state = 42, class_weight = 'balanced')
s2_clf = SVC(kernel='linear', C=1000,gamma = 0.0001,random_state = 42, class_weight = 'balanced')
s3_clf = SVC(kernel='poly', C=1000,gamma = 0.0001,random_state = 42, class_weight = 'balanced')
s4_clf = SVC(kernel='linear', C=2000,gamma = 0.0001,random_state = 42, class_weight = 'balanced')

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)

```

The whole purpose of developing a machine learning algorithm is to come up with an approach, that can be used to predict/classify new data. Validation enables us to check how well the algorithm generalizes when facing new data. This way, we can make sure that we are not over- & under-fitting. One of the classical mistakes here is to use the same data for both training and test. By doing this, we become oblivious to the weaknesses of our algorithm against new data. 
In order to validate the results of each algorithm, I divided the data into 70% training & 30% test set and used train_test_split by running the algorithms 1000 times while changing the data in each group with each iteration. Since we have a small group of people to investigate and really limited number of pois in it, it makes sense to dynamically change the group distributions and observer results over many iterations.


```python
# Function to evalute and validate algorithms
def clf_evaluater(clf, features, labels, num_iters=1000, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing:')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print ''
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    return mean(precision), mean(recall)
```


```python
print 'Naive Bayes with 10 features:', 
print clf_evaluater(nb_clf, features_10, labels_10)
print ''
print 'Kmeans with 10 features:'
print clf_evaluater(k_clf, features_10, labels_10)
print ''
print 'SVC with 10 features with several different tunnings:'
print clf_evaluater(s1_clf, features_10, labels_10)
print ''
print 'Random Forest with 10 features:'
print clf_evaluater(rf_clf, features_10, labels_10)
```

    Naive Bayes with 10 features: GaussianNB(priors=None)
    
    Processing:....................................................................................................
    precision: 0.330594534108
    recall:    0.355322940948
    (0.33059453410827677, 0.35532294094794098)
    
    Kmeans with 10 features:
    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.001, verbose=0)
    
    Processing:....................................................................................................
    precision: 0.284216744192
    recall:    0.34596998557
    (0.2842167441917543, 0.34596998556998559)
    
    SVC with 10 features with several different tunnings:
    SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
      max_iter=-1, probability=False, random_state=42, shrinking=True,
      tol=0.001, verbose=False)
    
    Processing:....................................................................................................
    precision: 0.439533861168
    recall:    0.278250793651
    (0.43953386116807169, 0.27825079365079364)
    
    Random Forest with 10 features:
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=5, max_features='sqrt', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=10, n_jobs=1, oob_score=False, random_state=42,
                verbose=0, warm_start=False)
    
    Processing:....................................................................................................
    precision: 0.344693253968
    recall:    0.164303318903
    (0.34469325396825401, 0.16430331890331892)


| Algorithm     | Precision     | Recall|
| :------------:|:-------------:|:-----:|
| Naive Bayes   | 0.331         | 0.355 |
| Kmeans        | 0.284         | 0.346 |
| SVC           | 0.440         | 0.278 |
| Random Forest | 0.335         | 0.164 |

As next, I would like to test manual tuning in SVC and see if I can improve my results. First I would like to test different kernels.


```python
print 'SVC with rbf kernel:'
print clf_evaluater(s1_clf, features_10, labels_10)
print ''
print 'SVC with linear kernel:'
print clf_evaluater(s2_clf, features_10, labels_10)
print ''
print 'SVC with poly kernel'
print clf_evaluater(s3_clf, features_10, labels_10)
```

    SVC with rbf kernel:
    SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
      max_iter=-1, probability=False, random_state=42, shrinking=True,
      tol=0.001, verbose=False)
    
    Processing:....................................................................................................
    precision: 0.436457683736
    recall:    0.284598412698
    (0.4364576837356709, 0.28459841269841274)
    
    SVC with linear kernel:
    SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape=None, degree=3, gamma=0.0001, kernel='linear',
      max_iter=-1, probability=False, random_state=42, shrinking=True,
      tol=0.001, verbose=False)
    
    Processing:....................................................................................................
    precision: 0.263260175179
    recall:    0.564141341991
    (0.26326017517876449, 0.56414134199134192)
    
    SVC with poly kernel
    SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape=None, degree=3, gamma=0.0001, kernel='poly',
      max_iter=-1, probability=False, random_state=42, shrinking=True,
      tol=0.001, verbose=False)
    
    Processing:....................................................................................................
    precision: 0.109877906977
    recall:    0.491583333333
    (0.10987790697674418, 0.49158333333333332)


| Algorithm     | Precision     | Recall|
| :------------:|:-------------:|:-----:|
| SVC (rbf)     | 0.436         | 0.284 |
| SVC (linear)  | 0.263         | 0.564 |
| SVC (poly)    | 0.110         | 0.491 |

The linear and poly kernels seem to increase the recall significantly while losing dramaticcaly on the precision. 

As next I would like to see what happens if I decrease the number of features I use to train my classifier. So I will use 3 features with same classifier. 


```python
print 'Naive Bayes with 3 features:', 
print clf_evaluater(nb_clf, features_3, labels_3)
print ''
print 'Kmeans with 3 features:'
print clf_evaluater(k_clf, features_3, labels_3)
print ''
print 'SVC with 3 features with several different tunnings:'
print clf_evaluater(s1_clf, features_3, labels_3)
print ''
print 'Random Forest with 3 features:'
print clf_evaluater(rf_clf, features_3, labels_3)
```

    Naive Bayes with 3 features: GaussianNB(priors=None)
    
    Processing:....................................................................................................
    precision: 0.491663347763
    recall:    0.33456046176
    (0.4916633477633478, 0.33456046176046172)
    
    Kmeans with 3 features:
    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.001, verbose=0)
    
    Processing:....................................................................................................
    precision: 0.565230743105
    recall:    0.314436385836
    (0.56523074310502719, 0.31443638583638583)
    
    SVC with 3 features with several different tunnings:
    SVC(C=1000, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
      max_iter=-1, probability=False, random_state=42, shrinking=True,
      tol=0.001, verbose=False)
    
    Processing:....................................................................................................
    precision: 0.516745618665
    recall:    0.180556060606
    (0.51674561866480373, 0.18055606060606061)
    
    Random Forest with 3 features:
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=5, max_features='sqrt', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=10, n_jobs=1, oob_score=False, random_state=42,
                verbose=0, warm_start=False)
    
    Processing:....................................................................................................
    precision: 0.504331890332
    recall:    0.246328571429
    (0.5043318903318903, 0.24632857142857142)


| Algorithm     | Precision     | Recall|
| :------------:|:-------------:|:-----:|
| Naive Bayes   | 0.492         | 0.335 |
| Kmeans        | 0.565         | 0.314 |
| SVC           | 0.517         | 0.181 |
| Random Forest | 0.504         | 0.246 |

So, moving from 10 to 3 features, I managed to improve my precision but lost on recall. Based on precision and recall without further tunning, naive bayes and Kmeans seem to be the good options here.

## Evaluation

In order to evaluate algorithms, I used recall and precision as my 2 metrics. For Naive Bayes I then got 49% precision and 34% recall. 
Precision shows the predictive capability of an algorithm. It is calculated as true positive / (true positive + false positive). In my case, it would mean from 100 people who are predicted to be pois, 45 of them will be truly pois. 
Recall on the other hand shows the capability of an algorithm to find pois. It is calculated as true positive / (true positive + false negative) . With 32% recall, my algorithm finds 32% of the pois in its prediction.
