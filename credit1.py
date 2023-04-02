Python 3.11.1 (tags/v3.11.1:a7a450f, Dec  6 2022, 19:58:39) [MSC v.1934 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
data = pd.read_csv('C:\\Users\\acer\\Desktop\\vrtul\\creditcard.csv')
pd.options.display.max_columns = None
data.head()
   Time        V1        V2        V3        V4        V5        V6        V7  \
0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   

         V8        V9       V10       V11       V12       V13       V14  \
0  0.098698  0.363787  0.090794 -0.551600 -0.617801 -0.991390 -0.311169   
1  0.085102 -0.255425 -0.166974  1.612727  1.065235  0.489095 -0.143772   
2  0.247676 -1.514654  0.207643  0.624501  0.066084  0.717293 -0.165946   
3  0.377436 -1.387024 -0.054952 -0.226487  0.178228  0.507757 -0.287924   
4 -0.270533  0.817739  0.753074 -0.822843  0.538196  1.345852 -1.119670   

        V15       V16       V17       V18       V19       V20       V21  \
0  1.468177 -0.470401  0.207971  0.025791  0.403993  0.251412 -0.018307   
1  0.635558  0.463917 -0.114805 -0.183361 -0.145783 -0.069083 -0.225775   
2  2.345865 -2.890083  1.109969 -0.121359 -2.261857  0.524980  0.247998   
3 -0.631418 -1.059647 -0.684093  1.965775 -1.232622 -0.208038 -0.108300   
4  0.175121 -0.451449 -0.237033 -0.038195  0.803487  0.408542 -0.009431   

        V22       V23       V24       V25       V26       V27       V28  \
0  0.277838 -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053   
1 -0.638672  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724   
2  0.771679  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752   
3  0.005274 -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458   
4  0.798278 -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   

   Amount  Class  
0  149.62      0  
1    2.69      0  
2  378.66      0  
3  123.50      0  
4   69.99      0  
data.tail()
            Time         V1         V2        V3        V4        V5  \
284802  172786.0 -11.881118  10.071785 -9.834783 -2.066656 -5.364473   
284803  172787.0  -0.732789  -0.055080  2.035030 -0.738589  0.868229   
284804  172788.0   1.919565  -0.301254 -3.249640 -0.557828  2.630515   
284805  172788.0  -0.240440   0.530483  0.702510  0.689799 -0.377961   
284806  172792.0  -0.533413  -0.189733  0.703337 -0.506271 -0.012546   

              V6        V7        V8        V9       V10       V11       V12  \
284802 -2.606837 -4.918215  7.305334  1.914428  4.356170 -1.593105  2.711941   
284803  1.058415  0.024330  0.294869  0.584800 -0.975926 -0.150189  0.915802   
284804  3.031260 -0.296827  0.708417  0.432454 -0.484782  0.411614  0.063119   
284805  0.623708 -0.686180  0.679145  0.392087 -0.399126 -1.933849 -0.962886   
284806 -0.649617  1.577006 -0.414650  0.486180 -0.915427 -1.040458 -0.031513   

             V13       V14       V15       V16       V17       V18       V19  \
284802 -0.689256  4.626942 -0.924459  1.107641  1.991691  0.510632 -0.682920   
284803  1.214756 -0.675143  1.164931 -0.711757 -0.025693 -1.221179 -1.545556   
284804 -0.183699 -0.510602  1.329284  0.140716  0.313502  0.395652 -0.577252   
284805 -1.042082  0.449624  1.962563 -0.608577  0.509928  1.113981  2.897849   
284806 -0.188093 -0.084316  0.041333 -0.302620 -0.660377  0.167430 -0.256117   

             V20       V21       V22       V23       V24       V25       V26  \
284802  1.475829  0.213454  0.111864  1.014480 -0.509348  1.436807  0.250034   
284803  0.059616  0.214205  0.924384  0.012463 -1.016226 -0.606624 -0.395255   
284804  0.001396  0.232045  0.578229 -0.037501  0.640134  0.265745 -0.087371   
284805  0.127434  0.265245  0.800049 -0.163298  0.123205 -0.569159  0.546668   
284806  0.382948  0.261057  0.643078  0.376777  0.008797 -0.473649 -0.818267   

             V27       V28  Amount  Class  
284802  0.943651  0.823731    0.77      0  
284803  0.068472 -0.053527   24.79      0  
284804  0.004455 -0.026561   67.88      0  
284805  0.108821  0.104533   10.00      0  
284806 -0.002415  0.013649  217.00      0  
data.shape
(284807, 31)
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   Time    284807 non-null  float64
 1   V1      284807 non-null  float64
 2   V2      284807 non-null  float64
 3   V3      284807 non-null  float64
 4   V4      284807 non-null  float64
 5   V5      284807 non-null  float64
 6   V6      284807 non-null  float64
 7   V7      284807 non-null  float64
 8   V8      284807 non-null  float64
 9   V9      284807 non-null  float64
 10  V10     284807 non-null  float64
 11  V11     284807 non-null  float64
 12  V12     284807 non-null  float64
 13  V13     284807 non-null  float64
 14  V14     284807 non-null  float64
 15  V15     284807 non-null  float64
 16  V16     284807 non-null  float64
 17  V17     284807 non-null  float64
 18  V18     284807 non-null  float64
 19  V19     284807 non-null  float64
 20  V20     284807 non-null  float64
 21  V21     284807 non-null  float64
 22  V22     284807 non-null  float64
 23  V23     284807 non-null  float64
 24  V24     284807 non-null  float64
 25  V25     284807 non-null  float64
 26  V26     284807 non-null  float64
 27  V27     284807 non-null  float64
 28  V28     284807 non-null  float64
 29  Amount  284807 non-null  float64
 30  Class   284807 non-null  int64  
dtypes: float64(30), int64(1)
memory usage: 67.4 MB
data.isnull()

data.isnull().sum()
Time      0
V1        0
V2        0
V3        0
V4        0
V5        0
V6        0
V7        0
V8        0
V9        0
V10       0
V11       0
V12       0
V13       0
V14       0
V15       0
V16       0
V17       0
V18       0
V19       0
V20       0
V21       0
V22       0
V23       0
V24       0
V25       0
V26       0
V27       0
V28       0
Amount    0
Class     0
dtype: int64
data.head()
   Time        V1        V2        V3        V4        V5        V6        V7  \
0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   

         V8        V9       V10       V11       V12       V13       V14  \
0  0.098698  0.363787  0.090794 -0.551600 -0.617801 -0.991390 -0.311169   
1  0.085102 -0.255425 -0.166974  1.612727  1.065235  0.489095 -0.143772   
2  0.247676 -1.514654  0.207643  0.624501  0.066084  0.717293 -0.165946   
3  0.377436 -1.387024 -0.054952 -0.226487  0.178228  0.507757 -0.287924   
4 -0.270533  0.817739  0.753074 -0.822843  0.538196  1.345852 -1.119670   

        V15       V16       V17       V18       V19       V20       V21  \
0  1.468177 -0.470401  0.207971  0.025791  0.403993  0.251412 -0.018307   
1  0.635558  0.463917 -0.114805 -0.183361 -0.145783 -0.069083 -0.225775   
2  2.345865 -2.890083  1.109969 -0.121359 -2.261857  0.524980  0.247998   
3 -0.631418 -1.059647 -0.684093  1.965775 -1.232622 -0.208038 -0.108300   
4  0.175121 -0.451449 -0.237033 -0.038195  0.803487  0.408542 -0.009431   

        V22       V23       V24       V25       V26       V27       V28  \
0  0.277838 -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053   
1 -0.638672  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724   
2  0.771679  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752   
3  0.005274 -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458   
4  0.798278 -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   

   Amount  Class  
0  149.62      0  
1    2.69      0  
2  378.66      0  
3  123.50      0  
4   69.99      0  
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data['Amount']=sc.fit_transform(data['Amount'])
Traceback (most recent call last):
  File "<pyshell#12>", line 1, in <module>
    data['Amount']=sc.fit_transform(data['Amount'])
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\utils\_set_output.py", line 142, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\utils\_set_output.py", line 142, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\base.py", line 848, in fit_transform
    return self.fit(X, **fit_params).transform(X)
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\preprocessing\_data.py", line 824, in fit
    return self.partial_fit(X, y, sample_weight)
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\preprocessing\_data.py", line 861, in partial_fit
    X = self._validate_data(
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\base.py", line 535, in _validate_data
    X = check_array(X, input_name="X", **check_params)
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\utils\validation.py", line 900, in check_array
    raise ValueError(
ValueError: Expected 2D array, got 1D array instead:
array=[149.62   2.69 378.66 ...  67.88  10.   217.  ].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
data['Amount']=sc.fit_transform(pd.DataFrame(data['Amount']))
data.head()
   Time        V1        V2        V3        V4        V5        V6        V7  \
0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   

         V8        V9       V10       V11       V12       V13       V14  \
0  0.098698  0.363787  0.090794 -0.551600 -0.617801 -0.991390 -0.311169   
1  0.085102 -0.255425 -0.166974  1.612727  1.065235  0.489095 -0.143772   
2  0.247676 -1.514654  0.207643  0.624501  0.066084  0.717293 -0.165946   
3  0.377436 -1.387024 -0.054952 -0.226487  0.178228  0.507757 -0.287924   
4 -0.270533  0.817739  0.753074 -0.822843  0.538196  1.345852 -1.119670   

        V15       V16       V17       V18       V19       V20       V21  \
0  1.468177 -0.470401  0.207971  0.025791  0.403993  0.251412 -0.018307   
1  0.635558  0.463917 -0.114805 -0.183361 -0.145783 -0.069083 -0.225775   
2  2.345865 -2.890083  1.109969 -0.121359 -2.261857  0.524980  0.247998   
3 -0.631418 -1.059647 -0.684093  1.965775 -1.232622 -0.208038 -0.108300   
4  0.175121 -0.451449 -0.237033 -0.038195  0.803487  0.408542 -0.009431   

        V22       V23       V24       V25       V26       V27       V28  \
0  0.277838 -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053   
1 -0.638672  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724   
2  0.771679  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752   
3  0.005274 -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458   
4  0.798278 -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   

     Amount  Class  
0  0.244964      0  
1 -0.342475      0  
2  1.160686      0  
3  0.140534      0  
4 -0.073403      0  
data = data.drop(['Time'],axis=1)
data.head()
         V1        V2        V3        V4        V5        V6        V7  \
0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
1  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
2 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
3 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
4 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   

         V8        V9       V10       V11       V12       V13       V14  \
0  0.098698  0.363787  0.090794 -0.551600 -0.617801 -0.991390 -0.311169   
1  0.085102 -0.255425 -0.166974  1.612727  1.065235  0.489095 -0.143772   
2  0.247676 -1.514654  0.207643  0.624501  0.066084  0.717293 -0.165946   
3  0.377436 -1.387024 -0.054952 -0.226487  0.178228  0.507757 -0.287924   
4 -0.270533  0.817739  0.753074 -0.822843  0.538196  1.345852 -1.119670   

        V15       V16       V17       V18       V19       V20       V21  \
0  1.468177 -0.470401  0.207971  0.025791  0.403993  0.251412 -0.018307   
1  0.635558  0.463917 -0.114805 -0.183361 -0.145783 -0.069083 -0.225775   
2  2.345865 -2.890083  1.109969 -0.121359 -2.261857  0.524980  0.247998   
3 -0.631418 -1.059647 -0.684093  1.965775 -1.232622 -0.208038 -0.108300   
4  0.175121 -0.451449 -0.237033 -0.038195  0.803487  0.408542 -0.009431   

        V22       V23       V24       V25       V26       V27       V28  \
0  0.277838 -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053   
1 -0.638672  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724   
2  0.771679  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752   
3  0.005274 -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458   
4  0.798278 -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   

     Amount  Class  
0  0.244964      0  
1 -0.342475      0  
2  1.160686      0  
3  0.140534      0  
4 -0.073403      0  
data.shape
(284807, 30)
data.duplicated().any()
True
data = data.drop_duplicates()
data.shape
(275663, 30)
284807-275663
9144
# Not handeling Imbalanced
data['Class'].value_counts()
0    275190
1       473
Name: Class, dtype: int64
import seaborn as sns
sns.countplot(data['Class'])
<AxesSubplot: ylabel='count'>
plt.show()
Traceback (most recent call last):
  File "<pyshell#26>", line 1, in <module>
    plt.show()
NameError: name 'plt' is not defined
import matplotlib.pyplot as plt
plt.show()
# Store Feature Matrix in X And Response(target) in vector y
x = data.drop('Class',axis=1)
y = data['Class']
# Spliting the Dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
# Logistic Regression
from sklearn.linear_model import LogisticRegression
log = 
KeyboardInterrupt
log = LogisticRegression()
log.fit(x_train,y_train)
LogisticRegression()
y_pred1 = log.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score()
Traceback (most recent call last):
  File "<pyshell#41>", line 1, in <module>
    accuracy_score()
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\utils\_param_validation.py", line 175, in wrapper
    params = func_sig.bind(*args, **kwargs)
  File "C:\Program Files\Python311\Lib\inspect.py", line 3210, in bind
    return self._bind(args, kwargs)
  File "C:\Program Files\Python311\Lib\inspect.py", line 3125, in _bind
    raise TypeError(msg) from None
TypeError: missing a required argument: 'y_true'
accuracy_score(y_test,y_pred1)
0.9992200678359603
from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y_test,y_pred1)
0.8870967741935484
recall_score(y_test,y_pred1)
0.6043956043956044
f1_score(y_test,y_pred1)
0.718954248366013
# Handeling imbalanced dataset
# Undersampling & Oversampling
## Undersampling
normal = data[data['Class']==0]
fraund = data[data['Class']==1]
normal.shape
(275190, 30)
fraud = data[data['Class']==1]
fraud.shape
(473, 30)
normal_sample=normal.sample(n=473)
normal_sample.shape
(473, 30)
new_data = pd.concat([normal_sample,fraud])
new_data['Class'].value_counts()
0    473
1    473
Name: Class, dtype: int64
new_data.head()
              V1        V2        V3        V4        V5        V6        V7  \
127664  1.141395 -1.547758  0.187315 -1.430198 -1.281610  0.218237 -1.001510   
14306   0.965830 -0.301032  1.191189  1.472875 -0.867514  0.321859 -0.553332   
230917 -1.265774  0.620914  1.357685 -0.759847  0.733310  0.847171  0.169572   
89565  -2.245474  1.945903  0.411740 -0.181944 -0.096932  0.214475 -0.993264   
242518  2.084044 -1.321658 -0.468818 -1.119615 -0.899179  0.592923 -1.426870   

              V8        V9       V10       V11       V12       V13       V14  \
127664  0.167342 -2.054150  1.494944  1.552775 -0.460263 -0.434255  0.180894   
14306   0.228511  0.846297 -0.217440 -0.616316  0.438720 -0.174471 -0.278266   
230917  0.483178 -0.190163 -0.786702  0.124412  0.897609  0.964494 -0.077338   
89565  -3.648722 -0.150863  0.155472  1.050647  0.684450 -0.324456 -0.001716   
242518  0.197737  0.253748  0.800727 -0.323205  0.118261  0.941918 -0.746371   

             V15       V16       V17       V18       V19       V20       V21  \
127664  0.497057 -0.756027  0.871826 -0.339620 -0.716369 -0.169869 -0.033777   
14306   0.422493 -0.153453  0.029224 -0.479696 -0.645606 -0.032983  0.005742   
230917  0.066773  0.382861 -0.896846  0.670516  1.034555 -0.013526 -0.195605   
89565   0.207141  0.784794  0.013558  0.572483  0.302495 -1.021706  3.492469   
242518 -0.419361  1.977835 -0.703859 -0.183060  1.216129  0.159542  0.212646   

             V22       V23       V24       V25       V26       V27       V28  \
127664 -0.012322 -0.029518 -0.325359  0.153970 -0.169822  0.025710  0.023736   
14306   0.078116 -0.033245  0.081377  0.302088 -0.370742  0.072502  0.047205   
230917 -0.428331 -0.003625 -0.228330  0.491035 -0.437262 -0.294585 -0.265380   
89565  -1.908264  0.768304 -0.131714 -0.000678  0.090883  0.119842 -0.241319   
242518  0.602194  0.038560 -1.357126 -0.286057 -0.188339  0.029734 -0.049392   

          Amount  Class  
127664  0.222495      0  
14306  -0.026826      0  
230917 -0.293258      0  
89565  -0.345313      0  
242518 -0.113344      0  
x = data.drop('Class',axis=1)
y = data['Class']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
accuracy_score(y_test,y_pred1)
0.9992200678359603
y_pred1 = log.predict(x_test)
accuracy_score(y_test,y_pred1)
0.9992200678359603
from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y_test,y_pred1)
0.8870967741935484
recall_score(y_test,y_pred1)
0.6043956043956044
normal = data[data['Class']==0]
fraund = data[data['Class']==1]
fraud = data[data['Class']==1]
normal_sample=normal.sample(n=473)
new_data = pd.concat([normal_sample,fraud])
x = data.drop('Class',axis=1)
y = data['Class']
x = new_data.drop('Class',axis=1)
y = new_data['Class']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train,y_train)
LogisticRegression()
y_pred1 = log.predict(x_test)
accuracy_score(y_test,y_pred1)
0.9368421052631579
from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y_test,y_pred1)
0.96875
recall_score(y_test,y_pred1)
0.9117647058823529
f1_score(y_test,y_pred1)
0.9393939393939394
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
DecisionTreeClassifier()
y_pred2 = dt.predict(x_test)
accuracy_score(y_test,y_pred2)
0.8947368421052632
precision_score(y_test,y_pred2)
0.8867924528301887
recall_score(y_test,y_pred2)
0.9215686274509803
f1_score(y_test,y_pred2)
0.9038461538461539
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
RandomForestClassifier()
y_pred3 = rf.predict(x_test)
accuracy_score(y_test,y_pred3)
0.9368421052631579
precision_score(y_test,y_pred3)
0.96875
recall_score(y_test,y_pred3)
0.9117647058823529
f1_score(y_test,y_pred3)
0.9393939393939394
final_data = pd.DataFrame({'Models':['LR','DT','RF'],"ACC":[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred2)*100,accuracy_score(y_test,y_pred3)*100]})
final_data
  Models        ACC
0     LR  93.684211
1     DT  89.473684
2     RF  93.684211
## Oversampling
import pandas as pd
data = pd.read_csv('C:\\Users\\acer\\Desktop\\vrtul\\creditcard.csv')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data['Amount']=sc.fit_transform(pd.DataFrame(data['Amount']))
data.head()
   Time        V1        V2        V3        V4        V5        V6        V7  \
0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   

         V8        V9       V10       V11       V12       V13       V14  \
0  0.098698  0.363787  0.090794 -0.551600 -0.617801 -0.991390 -0.311169   
1  0.085102 -0.255425 -0.166974  1.612727  1.065235  0.489095 -0.143772   
2  0.247676 -1.514654  0.207643  0.624501  0.066084  0.717293 -0.165946   
3  0.377436 -1.387024 -0.054952 -0.226487  0.178228  0.507757 -0.287924   
4 -0.270533  0.817739  0.753074 -0.822843  0.538196  1.345852 -1.119670   

        V15       V16       V17       V18       V19       V20       V21  \
0  1.468177 -0.470401  0.207971  0.025791  0.403993  0.251412 -0.018307   
1  0.635558  0.463917 -0.114805 -0.183361 -0.145783 -0.069083 -0.225775   
2  2.345865 -2.890083  1.109969 -0.121359 -2.261857  0.524980  0.247998   
3 -0.631418 -1.059647 -0.684093  1.965775 -1.232622 -0.208038 -0.108300   
4  0.175121 -0.451449 -0.237033 -0.038195  0.803487  0.408542 -0.009431   

        V22       V23       V24       V25       V26       V27       V28  \
0  0.277838 -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053   
1 -0.638672  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724   
2  0.771679  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752   
3  0.005274 -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458   
4  0.798278 -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   

     Amount  Class  
0  0.244964      0  
1 -0.342475      0  
2  1.160686      0  
3  0.140534      0  
4 -0.073403      0  
data = data.drop(['Time'],axis=1)
data.duplicated().any()
True
data.shape
(284807, 30)
data['Class'].value_counts()
0    284315
1       492
Name: Class, dtype: int64
x = data.drop('Class',axis=1)
y = data['Class']
x.shape
(284807, 29)
y.shape
(284807,)
from imblearn.over_sampling import SMOTE
x_res,y_res = SMOTE().fit_resample(x,y)
y_res.value.counts()
Traceback (most recent call last):
  File "<pyshell#127>", line 1, in <module>
    y_res.value.counts()
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\pandas\core\generic.py", line 5902, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'Series' object has no attribute 'value'. Did you mean: 'values'?
y_res.value_counts()
0    284315
1    284315
Name: Class, dtype: int64
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
x_train,x_test,y_train,y_test = train_test_split(x_res,y_res,test_size=0.20,random_state=42)
# Logistic Regression
log = LogisticRegression()
log.fit(x_train,y_train)
LogisticRegression()
y_pred1 = log.predict(x_test)
accuracy_score(y_test,y_pred1)
0.9466524805233631
precision_score(y_test,y_pred1)
0.9736161503395665
recall_score(y_test,y_pred1)
0.9184042403819151
f1_score(y_test,y_pred1)
0.9452046133976391
# Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
DecisionTreeClassifier()
y_pred2 = dt.predict(x_test)
accuracy_score(y_test,y_pred2)
0.9984875929866521
precision_score(y_test,y_pred2)
0.9977916432978127
recall_score(y_test,y_pred2)
0.9991926425161471
f1_score(y_test,y_pred2)
0.9984916514662551
# Random Forest Classifier
rf = RandomForestClassifier()
rf..fit(x_train,y_train)
SyntaxError: invalid syntax
rf.fit(x_train,y_train)
RandomForestClassifier()
y_pred3 = rf.predict(x_test)
accuracy_score(y_test,y_pred3)
0.9999120693596891
precision_score(y_test,y_pred3)
0.999824518302741
recall_score(y_test,y_pred3)
1.0
f1_score(y_test,y_pred3)
0.9999122514522385
final_data = pd.DataFrame({'Models':['LR','DT','RF'],"ACC":[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred2)*100,accuracy_score(y_test,y_pred3)*100]})
final_data
  Models        ACC
0     LR  94.665248
1     DT  99.848759
2     RF  99.991207
# Save the model
rf1 = RandomForestClassifier()
rf1.fit(x_res,y_res)
RandomForestClassifier()
import joblib
joblib.dump(rf1,"credit_card_model")
Traceback (most recent call last):
  File "<pyshell#163>", line 1, in <module>
    joblib.dump(rf1,"credit_card_model")
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\joblib\numpy_pickle.py", line 552, in dump
    with open(filename, 'wb') as f:
PermissionError: [Errno 13] Permission denied: 'credit_card_model'
>>> joblib.dump(rf1,"credit_card_model")
Traceback (most recent call last):
  File "<pyshell#164>", line 1, in <module>
    joblib.dump(rf1,"credit_card_model")
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\joblib\numpy_pickle.py", line 552, in dump
    with open(filename, 'wb') as f:
PermissionError: [Errno 13] Permission denied: 'credit_card_model'
>>> joblib.dump(rf1,"credit_card_model")
Traceback (most recent call last):
  File "<pyshell#165>", line 1, in <module>
    joblib.dump(rf1,"credit_card_model")
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\joblib\numpy_pickle.py", line 552, in dump
    with open(filename, 'wb') as f:
PermissionError: [Errno 13] Permission denied: 'credit_card_model'
>>> joblib.dump(rf1,"C:\\Users\\acer\\Desktop\\vrtul\\credit_card_model")
['C:\\Users\\acer\\Desktop\\vrtul\\credit_card_model']
>>> model = joblib.load("C:\\Users\\acer\\Desktop\\vrtul\\credit_card_model")
>>> model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

Warning (from warnings module):
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\base.py", line 409
    warnings.warn(
UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
array([0], dtype=int64)
>>> pred = model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

Warning (from warnings module):
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\base.py", line 409
    warnings.warn(
UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
>>> ans = model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

Warning (from warnings module):
  File "C:\Users\acer\AppData\Roaming\Python\Python311\site-packages\sklearn\base.py", line 409
    warnings.warn(
UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
>>> if ans == 0:
... print ("normal")
SyntaxError: expected an indented block after 'if' statement on line 1
>>> print("Normal") if ans==0 else print("Fraud")
Normal
