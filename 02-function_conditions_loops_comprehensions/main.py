# This is a sample Python script.
from typing import Any


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# FONKSİYONLAR, KOŞULLAR, DÖNGÜLER, COMPREHESİONS

#  FONKSİYONLAR ( Function)
#  KOŞULLAR     ( Conditions)
#  DÖNGÜLER     ( Loops)
#  Comprehesions


# Fonksiyonlar ( Functions)
    # Fonksiyon Okuryazarlığı

print("a", "b")

print("a", "b", sep="--")

help(print) #fonksiyon hakkında bilgi vermek

# Fonksiyon Tanımlama

    def calculate(x):
        print(x*2)

calculate(6)

# İki argümanlı/ parametreli bir fonksiyon tanımlayalım

    def summer(arg1, arg2):
        """
        Parameters/ Args
        :param arg1: int, float

        :param arg2: imt, float

        :return:

        :examples: summer(4,6)
        """
        print(arg1+arg2)

        summer(5,6)

        help(summer)


#Fonksiyonların Statement/ Body Bölümü

# def function_name(parameters/arguments)
#       statements ( function body)


def say_hi():
    print("Merhaba")
    print("Hi")
    print("Hello")
    say_hi()

def say_hi(string):
    print(string)
    print("Hi")
    print("Hello")
    say_hi("Emin")


    def multiplication(a, b):
        c =  a * b
        print(c)

        multiplication(10,9)

# girilen deerleri bir liste içinde saklayan fonksiyon

list_stroe = []

def add_element(a,b):
    c =  a * b
    list_stroe.append(c)
    print(list_stroe)

    add_element(1,5)
    add_element(18,8)
    add_element(180,10)

# Ön Tanımlı Argümanlar/ Parametreler ( Default Parameters / Arguments)
def divide(a,b):
    print(a/b)


divide(1,2)

def divide(a,b=1):
    print(a/b)

divide(1)

def say_hi(string="Merhaba"):
    print((string))
    print("Hi")
    print("Hello")

    say_hi()

# Ne zaman fonksiyon yazma ihtiyacımız olur

#v varm, mousture, charge


def calculate(varm, moisture, charge):
    print((varm+moisture)/charge)

    calculate(98,12,78)

# Return: Fonksiyon Çıktılarını Girdi olarak kullanmak

def calculate(varm, moisture, charge):
    return (varm+moisture)/charge

a1 = calculate(98,12,78)
a1
a2 = calculate(98,12,78)*10
a2

def calculate(varm, moisture, charge):
    varm = varm*2
    moisture = moisture*2
    charge = charge*2
    output = (varm+moisture)/charge
    return varm, moisture, charge, output
type(calculate(98,12,78))

charge: int | Any
varm, moisture, charge, output = calculate(98,12,78)

 # Fonksiyon İçerisinden Fonksiyon Çağırmak

 def calculate( varm, moisture, charge):
    return int((varm+moisture)/charge)

 calculate(90,12,12)*10

 def  standardization ( a, p):
     return  a * 10 / 100 * p * p

 standardization(45, 1)


 def all_calculation(varm, moisture, charge, p):
     a = calculate(varm,moisture,charge)
     b = standardization(a, p)
     print(b *10)

     all_calculation(1,3,5,12)

# Lokal & Global Değişkenler ( Local & Global Variables

list_stroe = [1, 2]

def add_element(a, b):
    c = a * b
    list_stroe.append(c)
    print(list_stroe)

add_element(1,5)

# Koşullar ( Conditions
# True-False Hatırlayalım
  #  1 == 1
   # 1 == 2
# if
if 1 == 1:
    print("something")

if 1 == 2:
    print("something")

number = 11

if number == 10:
    print("number is 10")

number = 10

def number_check(number):
    if number ==  10:
        print("number is 10")

number_check(12)

# else

def number_check(number):
    if number ==  10:
        print("number is 10")
    else:
        print("number is not 10")

number_check(12)

# elif

def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10 ")

number_check(10)


# Dongüler

# for loop


students =  ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]

for student in students:
    print(student.upper())

salaries =  [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary*20/100+salary))

for salary in salaries:
    print(int(salary*30/100+salary))

for salary in salaries:
    print(int(salary*50/100+salary))

    def new_salary(salary, rate):
        return int(salary*rate/100+salary)

new_salary(2540,13)

for salary in salaries:
    if salary  >= 3000:
        print(new_salary(salary,10))
    else:
        print(new_salary(salary, 20))


# Uygulama - Mülakat Sorusu

# Amaç: Aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz

# before : " hi my name is john and i am learning python"
# after : " Hi mY NaMe iS JoHn aNd i aM LeArNiNg PyThOn"

range(len("miuul"))
range(0,5)
for i in range(0,5):
    print(i)
def alternating( string):
    new_string  = ""
    # girelen string'in indexlerinde gez.
    for string_index in range(len(string)):
        # index çift ise büyük harfe çevir
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
            #index tek ise küçük harfe çevir
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("merhaba ben emin python ogreniyorum")

# break & continue & while

salaries = [1000, 2000, 3000, 4000, 5000]

# break
for salary in salaries:
    if salary == 3000:
        break
    print(salary)
# continue
for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

# while

number = 1
while number < 5:
    print(number)
    number += 1

# Enummerate: Otomatik Counter/ Indexer ile for loop

students =  ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)


for index, student in enumerate(students):
    print(index,student)

    A = []
    B = []
for index, student in enumerate(students):
   if index % 2 == 0:
       A.append(student)
   else:
       B.append(student)

print(A)
print(B)

# Uygulama - mülakat sorusu
# divide_students fonksiyonu yazınız
# çift indexte yer alan öğrencileri bir listeye
# tek indexte yer alan öğrencileri bir listeye
# fakar bu iki liste tek bir liste olarak return olsun

students =  ["John", "Mark", "Venessa", "Mariam"]

def divide_students(students):
    groups = [[],[]]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)

    return groups

divide_students(students)

# alternating fonksiyonunun enumerate ile yazılması

def alternating_with_enumerate(string):
    new_string =  ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

    alternating_with_enumerate("merhaba ben python ogreniyom")

# zip

students = ["John", "Mark", "Venessa", "Mariam"]

departments =  ["mathematics", "statistics","physics","astronmy"]

ages = [ 23, 30, 26, 22]

list(zip(students,departments,ages))

# lambda, map, filter, reduce

def summer(a, b):
    return a + b

summer(1,3 )*9

new_sum = lambda a,b:a + b

new_sum(4,5)

# map


salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x* 20/100+x

new_salary(5000)


for salary in salaries:
    print(new_salary(salary))

list(map(new_salary,salaries))


# del new_sum
list(map(lambda x: x * 20 / 100 + x, salaries))

# filter

list_stroe = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

list(filter(lambda x: x % 2 == 0, list_stroe))


# reduce

from functools import reduce
list_stroe =  [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_stroe)

# Comprehensions

# List Comprehension

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))

null_list = []

for salary in salaries:
    if salary >  3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))

 [new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]

 [salary * 2 for salary in salaries]

[salary * 2 for salary in salaries if salary < 3000 ]

[salary * 2 for salary in salaries if salary < 3000 ]

[salary * 2  if salary < 3000 else salary * 0 for salary in salaries]

[new_salary(salary * 2)  if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]


students =  ["John", "Mark", "Venessa", "Maram"]

students_no =  ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]

[student.upper() if student not in students_no else student.lower() for student in students]

# Dict Comprehension

dictionary =  { 'a': 1,
                'b': 2,
                'c': 3,
                'd': 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{k: v ** 2 for (k, v) in dictionary.items()}

{k.upper(): v  for (k, v) in dictionary.items()}

{k.upper(): v*2  for (k, v) in dictionary.items()}


# Uygulama- Mülakat Sorusu

# Amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istenmektedir
# Keyler orjinal deerler valueler ise değiştirilmiş değerler olacaktır.

numbers =  range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

{n: n ** 2 for n in numbers if n % 2 == 0}

# List & Dict COmprehension Uygulamalar

# Bir Veri Setindeki değişken isimlerini değiştirmek

# before:
# [ 'total', 'speeding', 'alcohol', 'not_dictracted', 'no_previous', 'ins_premium', 'ins_loses', 'abbrev']
# [ 'TOTAL', 'SPEADING', 'ALCOHOL', 'NOT_DICTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSES', 'ABBREV']

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())

A = []

for col in df.columns:
    A.append(col.upper())

df.columns =  A

df = sns.load_dataset("car_crashes")
df.columns = [col.upper() for col in df.columns]

# İsminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG eklemek istiyoruz
# before:  # [ 'TOTAL', 'SPEADING', 'ALCOHOL', 'NOT_DICTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSES', 'ABBREV']
# after:   # [ 'NO_FLAG_TOTAL', 'NO_FLAG_SPEADING', 'NO_FLAG_ALCOHOL', 'NO_FLAG_NOT_DICTRACTED', 'NO_FLAG_NO_PREVIOUS', 'FLAG_INS_PREMIUM', 'FLAG_INS_LOSES', 'NO_FLAG_ABBREV']

[col for col in df.columns if "INS" in col]

["FLAG_"+ col for col in df.columns if "INS" in col]

["FLAG_"+ col  if "INS" in col else "NO_FLAG_"+ col for col in df.columns]

df.columns = ["FLAG_"+ col  if "INS" in col else "NO_FLAG_"+ col for col in df.columns]

# Amaç key'i string , valuesi aşağıdaki gibi bir liste olan sözlük oluşturmak
# Bu işlemi sadece sayısal değişkenler için yapmak istiyoruz
# Output:
# { 'total':['mean','min', 'max', 'var'] ,
# 'speeding':['mean','min', 'max', 'var'] ,
# 'alcohol':['mean','min', 'max', 'var'] ,
# 'not_dictracted':['mean','min', 'max', 'var'] ,
# 'no_previous':['mean','min', 'max', 'var'] ,
# 'ins_premium':['mean','min', 'max', 'var'] ,
# 'ins_loses':['mean','min', 'max', 'var'] }

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

num_cols =  [col for col in df.columns if df[col].dtype != "O"]
soz = {}
agg_list =  ["mean", "min", "max", "sum"]

for col in num_cols:
    soz[col] =  agg_list

#kısa yol

new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)

wages = [1000,2000,3000,4000,5000]
new_wages =  lambda  x: x*0.20+x
list(map(new_wages,wages))

students = ["Denise", "Arsem","Tony","Audrey"]
low =  lambda x: x[0].lower()
print(list(map(low,students)))

dictn = {"Denise": 10, "Arsen": 12, "Tony": 15, "Audrey": 17}
new_dict =  {k: v*2 + 3 for (k, v) in dictn.items()}
new_dict

numbers = range(1,10)
numsda = {n: n ** 2 for n in numbers if n %2 != 0}