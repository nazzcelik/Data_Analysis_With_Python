###################
# NUMPY
###################

import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

# with numpy:
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

###################################################
# Numpy Array'i Oluşturmak (Creating Numpy Arrays)
###################################################

import numpy as np

np.array([1, 2, 3, 4, 5])
np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3, 4))

###################################################
# Numpy Array Özellikleri (Attibutes of Numpy Arrays)
###################################################
import numpy as np

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)
a.ndim  # 1
a.shape  # (5L,)
a.size  # 5
a.dtype  # int(32)

####################################
# Yeniden Şekillendirme (Reshaping)
####################################
import numpy as np

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

# 2.Yol
ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)

####################################
# Index Seçimi (Index Selection)
####################################
import numpy as np

a = np.random.randint(10, size=10)
# array([0, 4, 3, 0, 9, 1, 8, 4, 8, 2])
a[0]  # 0
a[0:5]  # array([0, 4, 3, 0, 9])
a[0] = 999  # array([999,   4,   3,   0,   9,   1,   8,   4,   8,   2])

m = np.random.randint(10, size=(3, 5))
# array([[4, 2, 6, 6, 8],
#        [2, 2, 6, 0, 9],
#        [4, 1, 7, 7, 6]])

m[0, 0]  # 4
m[1, 1]  # 2

m[2, 3] = 999
# array([[  4,   2,   6,   6,   8],
#        [  2,   2,   6,   0,   9],
#        [  4,   1,   7, 999,   6]])

m[2, 3] = 2.9
# array([[4, 2, 6, 6, 8],
#        [2, 2, 6, 0, 9],
#        [4, 1, 7, 2, 6]])

m[:, 0]  # array([4, 2, 4])
m[1, :]  # array([2, 2, 6, 0, 9])
m[0:2, 0:3]  # array([[4, 2, 6],
#                     [2, 2, 6]])

################
# Fancy Index
################
import numpy as np

v = np.arange(0, 30, 3)
#  array([ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27])
v[1]  # 3
v[4]  # 12

catch = [1, 2, 3]
v[catch]  # array([3, 6, 9])

################################################
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
################################################
import numpy as np

# Bu Array'de 3'ten küçük olan sayıları seçme işlemi yapacağız.
v = np.array([1, 2, 3, 4, 5])

# KLASİK DÖNGÜ İLE
ab = []
for i in v:
    if i < 3:
        ab.append(i)

# NUMPY İLE
v < 3
v[v < 3]

#################################################
# Matematiksel İşlemler (Mathematical Operations)
#################################################
import numpy as np

v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)

############################################
# NumPy ile İki Bilinmeyenli Denklem Çözümü
############################################

# 5*x0 + x1 = 12
# x0 + 3x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)
