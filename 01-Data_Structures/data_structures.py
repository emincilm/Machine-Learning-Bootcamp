######################################
# VERİ YAPILARI ( DATA STRUCTURES)
######################################
# - Veri Yapılarına Giriş ve Hızlı Özet
# - Sayılar ( Numbers):  int, float, complex
# - Karakter Dizileri ( Strings) :  str
# - Boolean ( TRUE-FALSE): bool
# - Liste (List)
# - Sözlük (Dictionary)
# - Demet (Tuple)
# - Set #


######################################
# Veri Yapılarına Giriş ve Hızlı Özet
#####################################
 # Sayılar: Integer
 x = 46
 type(x)
 # Sayılar : Float
 y = 10.3
 type(y)
 # Sayılar: Complex
 z =  2j + 1
 type(z)

 # String
 xx =  "Hello ai era"
 type(xx)

 # Boolean

 True
 type (True)

# Liste

x =  ["btc" , "eth", "xrp"]
type(x)

# Sözlük

x = {"name": "Peter", "Age": 36}
type(x)
# Tuple
x = ("python", "ml", "ds")
type(x)

# Set
x = {"python", "ml", "ds"}
type(x)

# Not : Liste , tuple, set ve dictionary veri yapıları
# aynı zamanda  Python Collections(Arrays) olarak geçmektedir

######################################
# Sayular ( Numbers):  int, float, complex
######################################

a = 5
b = 10.5

a * 3
a / 7
a * b / 10
a ** 2

######################################
# Tipleri Değiştirmek
######################################

int(b)
float(a)

######################################
# Karakter Dizileri ( Strings)
######################################

print("John")
print('Jogn')

"John"

name = "John"

######################################
# Çok Satırlı Karakter Dizileri
######################################

long_str = """ Veri Yapıları: Hızlı Özet,
Sayılar(Numbers): int, float, complex,
Karakter Dizileri ( Strings): set,
List, Dictionary, Tuple, Set,
Boolen ( TRUE-FALSE): bool"""

######################################
# Karakter Dizilerinin Elamnlarına Erişmek
######################################

name
name[0]
name[3]
name[0:2]
long_str[0:10]
######################################
# String İçerisinde Karakter Sorgulamak
######################################

long_str
"veri" in long_str
"Veri" in long_str
"bool" in long_str

######################################
# String (Karakter Dizisi) Methodları
######################################

dir(int)
dir(str)

######################################
# String (Karakter Dizisi) Methodları
######################################
name = "John"
type(name)
type(len)

len(name)
######################################
# upper lower
######################################
"miuul".upper()
"MEHMET".lower()
######################################
# replace karakter değiştirme
######################################
hi = "Hello AI Era"
hi.replace("l","p")

######################################
# Split: böler
######################################
hi.split()
######################################
# strip: kırpar
######################################
" ofofof ".strip()
"ofofofo".strip("o")

######################################
# capitalize:: ilk harfi büyütür
######################################

"mehmet".capitalize()

"foo".startswith("f")



######################################
# Liste ( List)
######################################

# - Değiştirilebilir
# - Sıralıdır. Index işlemleri yapılabilir
# - Kapsayıcıdır.

notes = [1, 2, 3, 4]
type(notes)

names = [ "a", "b", "c", "d"]

not_name = [ 1, 2, 3, "a", "b", True, [1, 2, 3]]

not_name[0]
not_name[5]
not_name[6]
not_name[6][1]

type(not_name[5])
type(not_name[6][1])

notes[0] =  99

not_name[0:4]

######################################
# Liste Metodları ( List Methods)
######################################

dir(notes)

len(notes)
len(not_name)

######################################
# append: eleman ekler
######################################


notes.append(100)
notes
######################################
# pop: indexe göre siler
######################################

notes.pop(0)
notes

######################################
# insert: indexe göre ekler
######################################

notes.insert(2,99)
notes

######################################
# Sözlük ( Dictionary
######################################

# - Değiştirilebilir
# - Sırasız. (3.7 den sonra sıralı)
# - Kapsayıcıdır.

# key-value

dictionary = {"Reg": "Regression",
              "Log": "Logistic Regression",
              "Cart": "Classification and Reg"}
dictionary["Reg"]

dictionary = {"Reg": ["RMSE", 10],
              "Log": ["MSE", 20],
              "Cart": ["SSE",30]}
dictionary["Cart"][1]

######################################
# Key Sorgulama
######################################

"Reg" in dictionary
"Ysa" in dictionary

######################################
# Key'e göre value erişmek
######################################

dictionary["Reg"]
dictionary.get("Reg")

######################################
# Value Değiştirmek
######################################

dictionary["Reg"] =  ["YSA",10]

######################################
# tüm keylere erişmek ve valuelere erişmek
######################################

dictionary.keys()
dictionary.values()

######################################
# Tüm Çiftleri Tuple Halinde listeye çevirme
######################################

dictionary.items()

######################################
# Key- Value Değerini Güncellemek
######################################

dictionary.update({"Reg": 11})

######################################
# Yeni Key- Value Eklemek
######################################

dictionary.update({"RF":10})

######################################
# Demet (Tuple)
######################################

# - Değiştirilemez
# - Sıralıdır
# - Kapsayıcıdır.

t =  ("john", "mark", 1, 2)

type(t)

t[0]
t[0:3]
t =  list(t)
t[0] = 99
t = tuple(t)

######################################
# Set
######################################
# - Değiştirilebilir
# - Sırasız. + Eşsizdir
# - Kapsayıcıdır.

######################################
# difference(): İki kümenin farkı
######################################

set1 = set([1, 2, 3])
set2 = set([1, 3, 5])

# set1' de olup set2'de olmayanlar
set1.difference(set2)
# set2' de olup set1'de olmayanlar
set2.difference(set1)

######################################
# symetric_difference(): İki kümede de birbirlerine göre olmayanlar
######################################

set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

######################################
# intersection(): iki kümenin kesişimi
######################################
set1.intersection(set2)
set2.intersection(set1)

######################################
# union(): İki kümenin birleşimi
######################################

set1.union(set2)
set2.union(set1)

######################################
# isdisjoint(): iki kümenin kesişimi boş mu
######################################
set1.isdisjoint(set2)
set2.isdisjoint(set1)

######################################
# issubset(): Bir küme diğer kümenin alt kümesi mi
######################################

set1.issubset(set2)
set2.issubset(set1)

######################################
# issuperset(): Bir küme diğer kümenin alt kümesi mi
######################################

set1.issuperset(set2)
set2.issuperset(set1)