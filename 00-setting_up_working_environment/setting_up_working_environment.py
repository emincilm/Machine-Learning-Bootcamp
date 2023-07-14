###################################################
# Sayılar (Numbers) ve Karakter Dizileri(Strings) #
###################################################
#integer
9
#float
9.2
#double
9.2331

# karakterler yani stringlerde tırnak işareti var ama sayılarda tırnak işareti yoktur

print(9)
type(9)
type(9.2)
type("MRB")     # str çıkar
print("Hellow World")

print("Hellow AI ERA")

###################################################
# Atamalar ve Değişkenler ( Assignments & Variables) #
###################################################

a = 9


b = "hello ai era"

c = 10

e = a * c

d = c - a

print(d)
print(e)

###################################################
# Virtual Environment ( Sanal Ortam  & Package Management ( Paket Yöneticisi #
###################################################

# Sanal ortalmalrın listelenmesi
# conda env list

# Sanal ortam oluşturma:
# conda create -n myenv

# Sanal ortamı aktif etme:
# conda activate myenv

# Yüklü Paketlerin listelenmesi
# conda list

# Paket yükleme
# conda install numpy

# Aynı anda birden fazla paket yükleme
# conda install numpy scripy pandas

# Paket silme
# conda remove package_name

# Belirli bir versiona göre paket yükleme
# conda install numpy=1.20.1

# Paket yükseltme:
# conda upgrade conda

# Tüm paketlerin yükseltilmesi
# conda upgrade -all

# pip: pypi ( python package  index) paket yönetim aracı

# Paket yükleme:
# pip install paket_adi

# Paket yükleme versiyona göre
#  pip install pandas==2.0.2

#