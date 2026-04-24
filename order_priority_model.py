import pandas as pd
import pickle  

df= pd.read_csv('numeric_copy.csv', encoding="ISO-8859-1")       
# Prepare the input data for prediction
x=df[['Ship Mode','Shipping Cost','Segment']]
y=df['Order Priority']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
from imblearn.combine import SMOTEENN
smoteenn = SMOTEENN(sampling_strategy='auto', random_state=42)
x_res, y_res = smoteenn.fit_resample(x, y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_res,y_res,test_size=0.2,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =2, metric = 'minkowski', p =2)
classifier.fit(x_train, y_train)

with open('order_priority_model.pkl', 'wb') as file:
    pickle.dump(classifier, file)
    