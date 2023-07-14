# Görev 1:  Verilen değerlerin veri yapılarını inceleyiniz.
# Type() metodunu kullanınız.

x = 8  # int
type(x)
y = 3.2  # float
type(y)
z = 8j + 18  # complex
type(z)
a = "Hello World"  # str
type(a)
b = True  # bool
type(b)
c = 23 < 22  # bool
type(c)
l1 = [1, 2, 3, 4]  # list
type(l1)
d = {"Name": "Jale",
     "Age": 27,
     "Adress": "Downtwon"}  # dict
type(d)
t = ("Macine Learning", "Data Science")  # tuple
type(t)
s = {"Python", "Machine Learning", "Data Science"}  # set
type(s)

# Görev 2:  Verilen string ifadenin tüm harflerini büyük harfe çeviriniz.
# Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
# Beklenen çıktı:
# String metodlarını kullanınız.
text = "The goal is to turn data into information and information into insgiht"
text_buyuk = text.upper()
text_kelime = text_buyuk.split()

# Görev 3:  Verilen listeye aşağıdaki adımları uygulayınız.
# lst = ["D","A","T","A","S","C","I","E","N","C","E"]
# Adım1: Verilen listenin eleman sayısına bakınız.
# Adım2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
# Adım3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
# Adım4: Sekizinci indeksteki elemanı siliniz.
# Adım5: Yeni bir eleman ekleyiniz.
# Adım6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1:
eleman_sayisi = len(lst)
print(eleman_sayisi)

# Adım 2:
sifirinci_indeks, onuncu_indeks = lst[0], lst[10]
print(sifirinci_indeks, onuncu_indeks)

# Adım 3:
yeni_lst = lst[0:4]
print(yeni_lst)

# Adım 4:
lst.pop(8)
print(lst)

# Adım 5:
lst.append("Emins")
print(lst)

# Adım 6:
lst.insert(8, "N")
print(lst)

# Görev 4:  Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
# dict1 = {'Christian': ["America", 18],
#         'Daisy': ["England", 12],
#         'Antonio': ["Spain", 22],
#         'Dante': ["Italy", 25]}
# Adım1: Key underline erişiniz.
# Adım2: Value'lara erişiniz.
# Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
# Adım4: Key değeri Ahmet value değeri[Turkey,24] olan yeni bir değer ekleyiniz.
# Adım5: Antonio'yu dictionary'den siliniz.

dict1 = {'Christian': ["America", 18],
         'Daisy': ["England", 12],
         'Antonio': ["Spain", 22],
         'Dante': ["Italy", 25]}

# Adım 1:
keys = dict1.keys()
print(keys)

# Adım 2:
values = dict1.values()
print(values)

# Adım 3:
dict1['Daisy'][1] = 13
print(dict1)

# Adım 4:
dict1['Ahmet'] = ['Turkey', 24]
print(dict1)

# Adım 5:
dict1.pop('Antonio')
print(dict1)

# Görev 5:Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları
# ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız.
# Liste elemanlarına tek tek erişmeniz gerekmektedir.
# Her bir elemanın çift veya tek olma durumunu kontrol etmekiçin  % yapısını kullanabilirsiniz.

liste1 = [2, 13, 18, 93, 22]


def even_odd(liste):
    even_list = []
    odd_list = []
    for n in liste:
        if n % 2 == 0:
            even_list.append(n)
        else:
            odd_list.append(n)

    return even_list, odd_list


even_odd(liste1)

# Görev 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin
# isimleri bulunmaktadır. Sırasıylailk üç öğrenci mühendislik fakültesinin başarı sırasını
# temsil ederken son üçöğrencide tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.


ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

muh_fak = ogrenciler[:3]
tip_fak = ogrenciler[3:]


def okul_siralama(ogrenciler):

    for i, ogrenci in enumerate(ogrenciler, start=1):
        if i <= 3:
            print(f"Mühendislik fakültesi {i}. öğrenci: {ogrenci}")
        else:
            print(f"Tıp fakültesi {i - 3}. öğrenci: {ogrenci}")


okul_siralama(ogrenciler)

# Görev 7 Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu,
# kredisi ve kontenjan bilgileri yer almaktadır.
# Zip kullanarak ders bilgilerini bastırınız.

ders_kod = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

dersler = zip(kredi, ders_kod, kontenjan)

for kredi,ders_kod,kontenjan in dersler:
    print(f"Kredisi {kredi} olan {ders_kod} kodlu dersin kontenjanı {kontenjan} kişidir")

# Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise
# ortak elemanlarını eğerk apsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu
# tanımlamanız beklenmektedir.

kume1 = set(["data", 'pyhton'])
kume2 = set(["data", "function", "qcut", "lambda", "pyhton", "miuul"])

def kume_diff(kume1, kume2):
    if kume1.issuperset(kume2):
        ortak_elemans = kume1.intersection(kume2)
        print(ortak_elemans)
    else:
        fark = kume2.difference(kume1)
        print(fark)

kume_diff(kume1, kume2)


# List Comprehension Alıştırmalar
# Görev 1:  List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin
# isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
df.dtypes

["NUM_" + col.upper() if df[col].dtype != 'O' else col.upper() for col in df.columns]

# Görev 2: List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan
# değişkenlerin isimlerinin sonuna "FLAG" yazınız.

[col.upper() if "no" in col else col.upper() + "_FLAG" for col in df.columns]

# Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan
# değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if not(col in og_list)]
new_df = df[new_cols]
new_df.head()