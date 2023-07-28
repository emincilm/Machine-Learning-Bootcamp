import numpy as np

a = [ 1, 2, 3, 4]
b = [ 2, 3, 4, 5]

ab = []

for i in range(0, len(a)):
    ab.append((a[i] * b[i]))

a = np.array([1,2,3,4])
b = np.array([2,3,4,5])

a * b


# NumPy Array'i Oluşturmak ( Creating Numpy Arrays)

np.array([1, 2, 3, 4, 5])

type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype = int)

np.random.randint(0,10,size = 10)

np.random.normal(10, 4, (3,4))


# NumPy Array Özellikleri

a =  np.random.randint(10, size= 5)

a.ndim

a.shape

a.size

a.dtype


np.random.randint(1,10, size = 9)
np.random.randint(1,10, size = 9).reshape(3, 3)

ar =  np.random.randint(1,10, size = 9)
ar.reshape(3, 3)


# Index seçimi ( Index Selection)

a = np.random.randint(10, size = 10)
a[0]
a[0:5]
a[0] = 999

m = np.random.randint(10, size = (3, 5))
m[0, 0]
m[1, 1]
m[2, 3] = 999
m[2, 3] = 2.3


m[:, 0]

m[0:2, 0:3]

# Fancy Index

v =  np.arange(0, 30, 3)
v[3]

catch = [1, 2, 3]

v[catch]


# NumPy Koşullu İşlemler
v1 = np.arange(1,6,1)
v1 = np.array([1,2,3,4,5])

# klasik döngü ile
ab = []

for i in v1:
    if i < 3:
        ab.append(i)


# NumPy ile

v1 < 3

v[v1 < 3]
v[v1 > 3]
v[v1 != 3]
v[v1 == 3]
v[v1 >= 3]

# Matematiksel işlemler
v1 = np.arange(1,6,1)
v1 / 5
v1 * 5 / 10
v1 ** 2
v1-1

np.subtract(v1,1)
np.add(v1,1)
np.mean(v1)
np.sum(v1)
np.min(v1)
np.max(v1)
np.var(v1)

v = np.subtract(v1, 1)

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])
np.linalg.solve(a, b)

a = np.array([2, 4, 6, 8])
a**2


# Pandas Series
import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)



# Your pandas-related code here
# For example:
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# Now you can work with 'pd' (pandas) functions and objects like 'df' (DataFrame).


# Veri Okuma ( Reading Data)


df = pd.read_csv("datasets\._advertising.csv")

df.head()


# Veriye Hızlı Bakış

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")

df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df['sex'].head()
df['sex'].head().value_counts()

# Pandasta Seçim İşlemleri
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df.drop(0,axis=0).head()

delete_indexes = [1, 3, 5, 7]

df.drop(delete_indexes, axis=0).head(10)
# kalıcı olması için tekrar atama yaparız yada inplace ayaparak kalıcı yaparız

# Değişkeni Indexe Çevirmek

df["age"].head()
df.age.head()

df.index

df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace= True).head()

df.index
df["age"] =  df.index

df.head()

df.reset_index().head()
df =  df.reset_index()
df.head()
# Değişken Üzerinde İşlemler

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head())

df[["age"]].head()
type(df[["age"]].head())

df[["age","alive"]]

col_names = [ "age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"]**2
df["age3"] = df["age"] / df ["age2"]

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()

df.loc[:, ~df.columns.str.contains("age")].head()

# iloc & loc

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# iloc : integer based selection
df.iloc[0:3]
df.iloc[0, 0]

# looc : label based selectioon
df.loc[0:3]

df.iloc[0:3, 0:3]
df.loc[0:3, "age"]

col_names = [ "age", "embarked", "alive"]
df.loc[0:3, col_names]

# Koşullu Seçim

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, ["age", "class"]].head()

df.loc[df["age"] > 50 & (df["sex"] == "male"), ["age", "class"]].head()

df["embark_town"].value_counts()
df_new = df.loc[df["age"] > 50 & (df["sex"] == "male") & ((df["embark_town"] == "Cherbourg")| (df["embark_town"] == "Southampton")),["age", "class","embark_town"]]

df_new['embark_town'].value_counts()

# Toplulaştırma ve Gruplama
# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table


import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "embark_town": "count"})

df.groupby(["sex", "embark_town"]).agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town","class"]).agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town","class"]).agg({
            "age": ["mean", "sum"],
            "survived": "mean",
            "sex": "count"})


# Pivot Table
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked")

df.pivot_table("survived", "sex", ["embarked", "class"])

df.head()

df["new_ae"] = pd.cut(df["age"], [0, 10, 18, 25, 49, 90])

df.pivot_table("survived", "sex", ["new_ae","class"])

pd.set_option('display.width', 500)

# Apply ve Lambda
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()


df["age2"] = df['age']*2
df["age3"] = df['age']*3

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

df.head()

df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x -x.mean()) / x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean())/col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.head()

# Birleştirme (Join) İşlemleri

import pandas as pd
import seaborn as sns

m =  np.random.randint(1, 30, size=(5, 3))

df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])

df2 = df1 + 99

pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index=True)

# Merge ile  Birleştirme

df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering','engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)

pd.merge(df1, df2, on="employees")

# Amaç: Her çalışanın müdürünün bilgisine erişmek istiyoruz

df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting','engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})

pd.merge(df3,df4)


series = pd.Series([1,2,3])
series**2

dict = { "Paris": [10], "Berlin": [20]}

pd.DataFrame(dict)

# Veri Görselleştirme: Matplotlib & seabon

# matplotlib

# Kategorik değişken: sütun grafik countplot bar
# Sayısal değişken: hist, boxplot


# Kategorik Değişken Görselleştirme
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()


df["sex"].value_counts().plot(kind='bar')
plt.show()

# Sayısal Değişken Görselleştirme

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show(block = True)

# Matplotlib'in Özellikleri
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# plot
x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show(block=True)

plt.plot(x, y, 'o')
plt.show(block=True)

# Marker

y =  np.array([13, 28, 11, 100])

plt.plot(y, marker = 'o')
plt.show(block=True)

plt.plot(y, marker = '*')
plt.show(block=True)

markers =  ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd']

# line
y =  np.array([13, 28, 11, 100])

plt.plot(y,linestyle="dashed")
plt.show(block=True)

plt.plot(y, linestyle="dotted")
plt.show(block=True)

plt.plot(y, linestyle="dashdot")
plt.show(block=True)

plt.plot(y, linestyle="dashdot", color="r")
plt.show(block=True)

# Multiple Lines

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show(block=True)

# Labels

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320])


# Başlık
plt.title("Bu ana başlık")

# X eksenini isimlendirme
plt.xlabel("X ekseni isimlendirmesi")
# Y eksenini isimlendirme
plt.ylabel("Y Ekseni isimlendirme")

plt.grid()
plt.plot(x, y)
plt.show(block=True)

# Subplots
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320])
plt.subplot(1,2,1)
plt.title("1")
plt(x,y)



# Seaborn

import pandas as pd
import seaborn as sns
from  matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show(block=True)

df['sex'].value_counts().plot(kind ="bar")
plt.show(block=True)

# Sayısal Deişken Görselleştirme
sns.boxplot(x=df["total_bill"])
plt.show(block=True)

df["total_bill"].hist()
plt.show(block=True)

# Gelişmiş Fonksiyonel Keşifçi Veri Analizi
# 1 Genel Resim
# 2 Kategorik Değişken Analizi
# 3 Sayısal Değişken Analizi
# 4 Hedef Değişken Analizi
# 5 Korelasyon Analizi

# 1 Genel Resim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head=5):
    print(" Shape")
    print(dataframe.shape)
    print("Types")
    print(dataframe.dtypes)
    print("Heaad")
    print(dataframe.head(head))
    print("Tail")
    print(dataframe.tail(head))
    print("NA")
    print(dataframe.isnull().sum())
    print("Quantiles")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.00, 1]).T)

check_df(df)

df = sns.load_dataset("tips")
check_df(df)

df = sns.load_dataset("flights")
check_df(df)

# Kategorik Değişken Analizi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()
df["embarked"].value_counts()
df["sex"].unique()
df["class"].nunique()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100*dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################3")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100*dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################3")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

df["adult_male"].astype(int)

##############################33
def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100*dataframe[col_name].value_counts() / len(dataframe)}))
        print("##############################3")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##############################3")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


cat_summary(df, "sex",plot=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)


########### 3 Sayısal Değişken Analizi

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[["age","fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"] ]

num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df,"age", plot=True)

for col in num_cols:
    num_summary(df,col,plot=True)


##### Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()
df.info()


def grab_col_names(dataframe, cat_th=10, car_th=30):
    """
    Veri setindei kategorik, numerik ve kategorik fakat kardinal deişkenlerin isimleridir
    :param dataframe:
    değişken isimleri alınmak istenen dataframedir
    :param cat_th:
    int, float
    numerik fakat kategorik olan değişkenler için sınıf eşik deeri
    :param car_th:
    kategorik fakat kardinal değişkenler için sınıf eşik değeri
    :return:
    cat_cols: list
    Kategorik değişken listei
    num_cols: list
    Numerik değişken listei
    cat_but_car: list
    kategorik görünümlü kardinal değişken listei

    Notess

    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100*dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################3")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Bonus

df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# 4 Hedef Değişken Analizi

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] =  df[col].astype((int))

def grab_col_names(dataframe, cat_th=10, car_th=30):
    """
    Veri setindei kategorik, numerik ve kategorik fakat kardinal deişkenlerin isimleridir
    :param dataframe:
    değişken isimleri alınmak istenen dataframedir
    :param cat_th:
    int, float
    numerik fakat kategorik olan değişkenler için sınıf eşik deeri
    :param car_th:
    kategorik fakat kardinal değişkenler için sınıf eşik değeri
    :return:
    cat_cols: list
    Kategorik değişken listei
    num_cols: list
    Numerik değişken listei
    cat_but_car: list
    kategorik görünümlü kardinal değişken listei

    Notess

    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

df["survived"].value_counts()
cat_summary(df, "survived")

# Hedef Değişkenin Kategorik Değişken ile Analizi

df.groupby("sex")["survived"].mean()

def target_cummary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"Target Mean": dataframe.groupby(categorical_col)[target].mean()}))

target_cummary_with_cat(df, "survived", "pclass")


for col in cat_cols:
    target_cummary_with_cat(df, "survived", col)

# Hedef Değişkeni Sayısal Değişken ile Analizi

df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age":"mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


# Korelasyon Analizi

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = pd.read_csv("datasets/._breast_cancer.csv")
df = df.iloc[:, 1:-1]
df.head

num_cols = [col for col in df.columns if df[col].dtype in [int,float]]

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, corr="RdBu")
plt.show(block=True)