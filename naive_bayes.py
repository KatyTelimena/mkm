#import bibliotek
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer #rozbija dataset wzdgledem nazwy kolumn, zaimplementuj cos tylko do num col a cos tylko do cat column
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


#naive bayes NB
path1 = "https://raw.githubusercontent.com/Kamil128/SDA_Excercise/main/data/employee/df2.csv"
path2 = "https://raw.githubusercontent.com/Kamil128/SDA_Excercise/main/data/employee/df1.csv"
path3 = "https://raw.githubusercontent.com/Kamil128/SDA_Excercise/main/data/employee/attrition.csv"

#przypisanie zmiennych do ścieżek
data1 = pd.read_csv(path1)
data2 = pd.read_csv(path2)
data3 = pd.read_csv(path3)

#łączenie tabel za pomocą join() i set_index()
table=data1.set_index('EmployeeNumber').join(data2.set_index('EmployeeNumber'))

#resetowanie indeksu za pomocą reset_index()
table.reset_index(inplace=True) #resetowanie indeksu, nie trzeba zapisywac do zmiennej z inplace=True

#usuwanie wartości NaN
table_drop=table.dropna()
table_drop.info()

# Usunięcie niepotrzebnych kolumn (z 1 wartością)

col_to_drop = ['Over18', 'StandardHours',"EmployeeCount"]
table_drop.drop(col_to_drop, axis=1, inplace=True)

# Definiowanie zmiennej zależnej y i zmiennych niezależnych x
X=table_drop.drop(['Attrition'],axis=1)
y=table_drop['Attrition']

#Podział na zbiór testowy i treningowy

X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2,
                                               shuffle=True,
                                               random_state=42)
#stratify=y (wartość domyślna)
#shuffle=True (wartość domyślna)

#podzial na atrubyty numeryczne i kategoryczne do pipeline'u
num_att=X_train.select_dtypes(include='number').columns
cat_att=X_train.select_dtypes(exclude='number').columns

##tworzenie pipelinu dla danych numerycznych z użyciem StandardScaler
num_pipeline=Pipeline(
    [
     ('std_scaler',StandardScaler())
    ]
)

#tworzenie pipelinu dla danych kategorycznych z użyciem OHE
cat_pipeline=Pipeline(
    [
     ('ohe',OneHotEncoder()),
    ]
)

#połączenie
cat_num_pipeline = ColumnTransformer( #do polaczenia dwoch kolumn
    [
     ('numerical',num_pipeline, num_att),
     ('categorical',cat_pipeline, cat_att)
    ]
)

#zapis do zmiennej
X_tr = cat_num_pipeline.fit_transform(X_train)
X_tr

#NAIWNY BAYES
naive_bayes= Pipeline(
    [
      ('model',MultinomialNB())
    ]
)

naive_bayes.fit(X_tr,y_train)
y_pred=naive_bayes.predict(X_test)