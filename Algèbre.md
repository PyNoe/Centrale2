# Entrainement. Oral Centrale 2 - Algèbre

### Exercice 1.

**Question 1.**

```python
import numpy as np

def chi(n) :
	A = np.zeros((n, n))
	for i in range(0, n):
		A[i, n-1] = 1/n
	for i in range(0, n-1):
		A[i+1, i] = 1
	return np.poly(A)

for i in range(2, 9):
	print(chi(i))
```

**Question 2.**

On conjecture :
$$\chi_n = X^n - \sum_{i=0}^{n-1}\dfrac{1}{n}X^i$$
**Question 3.**

```python
import numpy as np
import numpy.linalg as alg

def valeurspropres(n) :
	A = np.zeros((n, n))
	for i in range(0, n):
		A[i, n-1] = 1/n
	for i in range(0, n-1):
		A[i+1, i] = 1
	L = alg.eigvals(A)
	return [abs(x) for x in L]

for i in range(2, 9):
	print(valeurspropres(i))
```

On obtient :

```
[0.5, 1.0]
[0.577350269189626, 0.577350269189626, 1.0]
[0.6058295861882685, 0.6423840734792909, 0.6423840734792909, 1.0]
[0.6462441275530105, 0.6462441275530105, 0.6920195889335538, 0.6920195889335538, 1.0]
[0.6703320476030974, 0.6828225222964587, 0.6828225222964587, 0.7302499667488687, 0.7302499667488687, 1.0]
[0.6963069309595127, 0.6963069309595127, 0.7139063751575332, 0.7139063751575332, 0.9999999999999986, 0.7603420364683512, 0.7603420364683512]
[0.714537727167335, 0.7203689990737171, 0.7203689990737171, 0.7400629298553058, 0.7400629298553058, 0.7845466257669382, 0.7845466257669382, 1.0]
```

On remarque que 1 est valeur propre $A_n$ et que les autres valeurs propres ont un module strictement inférieur à 1.



### Exercice 2.

**Question 1.**

```python
import numpy as np
import numpy.linalg as alg

def M(a,b):
	M = np.array(([3*a-2*b, -6*a + 6*b +3], [a-b, -2*a + 3*b +1]))
	return M

def e(a,b):
	L = alg.eigvals(M(a,b))
	return round(abs(L[0]-L[1]),2)
```

**Question 2.**

```python
import numpy.random as rd

def compte(p):
	compteur = 0
	A = rd.geometric(p, 500)
	B = rd.geometric(p, 500)
	for i in range(len(A)):
		if e(A[i], B[i]) >= 0.1:
			compteur += 1
	return compteur
```

**Question 3 et 4**

```python
import matplotlib.pyplot as plt

def hasard(p):
	compteur = 0
	A = rd.geometric(p, 500)
	B = rd.geometric(p, 500)
	for i in range(len(A)):
		if e(A[i], B[i]) >= 0.1:
			compteur += 1
	return compteur

x = [i/100 for i in range(1,100)]
y = [hasard(j)/500 for j in x ]

def f(p):
	return (2-2*p+p**2)/(2-p)

P = np.linspace(0, 1, 256)

plt.plot(P, f(P), label="Equation conjecturée")
plt.plot(x,y, label="Simulations")
plt.legend()
plt.show()
```

On obtient :

<img src="/Users/noedaniel/Library/Application Support/CleanShot/media/media_paLEUognhY/CleanShot 2023-06-15 at 18.46.31@2x.png" alt="CleanShot 2023-06-15 at 18.46.31@2x" style="zoom:50%;" />

### Exercice 3.

**Question 1.**

```python
import numpy as np
import numpy.linalg as alg
import numpy.random as rd
import matplotlib.pyplot as plt

def faireA(n,a):
	A = np.zeros((n,n))
	for i in range(0, n-1):
		A[i+1,i] = a
		A[i, i+1] = 1/a
	return A
```

**Question 2.**

```python
def valpropres(n,a):
	A = faireA(n,a)
	return alg.eigvals(A)

for a in range(-2, 4):
	for n in range(3, 9):
		if a != 0:
			print("n = ",n," et a =",a, " >>", valpropres(n, a))
```

On obtient : 

```
n =  3  et a = -2  >> [-1.41421356e+00  9.24824453e-17  1.41421356e+00]
n =  4  et a = -2  >> [-1.61803399 -0.61803399  1.61803399  0.61803399]
n =  5  et a = -2  >> [-1.73205081e+00 -1.00000000e+00 -2.38375490e-16  1.73205081e+00
  1.00000000e+00]
n =  6  et a = -2  >> [-1.80193774 -1.2469796  -0.44504187  1.80193774  0.44504187  1.2469796 ]
n =  7  et a = -2  >> [-1.84775907e+00 -1.41421356e+00 -7.65366865e-01  3.42046011e-16
  1.84775907e+00  1.41421356e+00  7.65366865e-01]
n =  8  et a = -2  >> [-1.87938524 -1.53208889 -1.         -0.34729636  0.34729636  1.87938524
  1.53208889  1.        ]
n =  3  et a = -1  >> [-1.41421356e+00  9.24824453e-17  1.41421356e+00]
n =  4  et a = -1  >> [-1.61803399 -0.61803399  1.61803399  0.61803399]
n =  5  et a = -1  >> [ 1.73205081e+00 -1.73205081e+00 -1.00000000e+00 -7.27147645e-19
  1.00000000e+00]
n =  6  et a = -1  >> [-1.80193774 -1.2469796  -0.44504187  1.80193774  0.44504187  1.2469796 ]
n =  7  et a = -1  >> [-1.84775907e+00 -1.41421356e+00 -7.65366865e-01  4.74041766e-16
  1.84775907e+00  1.41421356e+00  7.65366865e-01]
n =  8  et a = -1  >> [-1.87938524 -1.53208889 -1.         -0.34729636  0.34729636  1.87938524
  1.53208889  1.        ]
n =  3  et a = 1  >> [-1.41421356e+00  9.24824453e-17  1.41421356e+00]
n =  4  et a = 1  >> [-1.61803399 -0.61803399  1.61803399  0.61803399]
n =  5  et a = 1  >> [ 1.73205081e+00 -1.73205081e+00 -1.00000000e+00 -7.27147645e-19
  1.00000000e+00]
n =  6  et a = 1  >> [-1.80193774 -1.2469796  -0.44504187  1.80193774  0.44504187  1.2469796 ]
n =  7  et a = 1  >> [-1.84775907e+00 -1.41421356e+00 -7.65366865e-01  4.74041766e-16
  1.84775907e+00  1.41421356e+00  7.65366865e-01]
n =  8  et a = 1  >> [-1.87938524 -1.53208889 -1.         -0.34729636  0.34729636  1.87938524
  1.53208889  1.        ]
n =  3  et a = 2  >> [-1.41421356e+00  9.24824453e-17  1.41421356e+00]
n =  4  et a = 2  >> [-1.61803399 -0.61803399  1.61803399  0.61803399]
n =  5  et a = 2  >> [-1.73205081e+00 -1.00000000e+00 -2.38375490e-16  1.73205081e+00
  1.00000000e+00]
n =  6  et a = 2  >> [-1.80193774 -1.2469796  -0.44504187  1.80193774  0.44504187  1.2469796 ]
n =  7  et a = 2  >> [-1.84775907e+00 -1.41421356e+00 -7.65366865e-01  3.42046011e-16
  1.84775907e+00  1.41421356e+00  7.65366865e-01]
n =  8  et a = 2  >> [-1.87938524 -1.53208889 -1.         -0.34729636  0.34729636  1.87938524
  1.53208889  1.        ]
n =  3  et a = 3  >> [-1.41421356e+00  5.89805982e-17  1.41421356e+00]
n =  4  et a = 3  >> [-1.61803399 -0.61803399  1.61803399  0.61803399]
n =  5  et a = 3  >> [-1.73205081e+00 -1.00000000e+00 -1.27028214e-16  1.73205081e+00
  1.00000000e+00]
n =  6  et a = 3  >> [-1.80193774 -1.2469796  -0.44504187  1.80193774  0.44504187  1.2469796 ]
n =  7  et a = 3  >> [-1.84775907e+00 -1.41421356e+00 -7.65366865e-01  1.96194186e-16
  1.84775907e+00  1.41421356e+00  7.65366865e-01]
n =  8  et a = 3  >> [-1.87938524 -1.53208889 -1.         -0.34729636  1.87938524  1.53208889
  0.34729636  1.        ]
```

On remarque que l'on a toujours $n$ valeurs propres distinctes réelles, donc le polynôme caractéristique est scindé à racines simples dans $\mathbb{R}[X]$, donc la matrice est diagonalisable. De plus **les valeurs propres sont indépendantes de $a$.**

**Question 3.**

```python
from numpy.polynom import Polynomial

p1 = Polynomial([0, 1])
p2 = Polynomial([-1, 0, 1])

P = [p1, p2]
for j in range(2, 8):
	pj = (p1)*P[j-1] - P[j-2]
	P.append(pj)
	
for p in P:
	print(p.coef)
```

On obtient :

```
[0. 1.]
[-1.  0.  1.]
[ 0. -2.  0.  1.]
[ 1.  0. -3.  0.  1.]
[ 0.  3.  0. -4.  0.  1.]
[-1.  0.  6.  0. -5.  0.  1.]
[ 0. -4.  0. 10.  0. -6.  0.  1.]
[  1.   0. -10.   0.  15.   0.  -7.   0.   1.]
```

Soit donc :
$$P_1 = X$$ $$P_2 = X^2 - 1$$ $$P_3 = X^3 - 2X$$ $$P_4 =X^4 - 3X^2 +1$$ $$P_5 = X^5 - 4X^3 + 3X$$ $$P_6 = X^6 - 5X^4 + 6X^2 - 1$$ $$P_7 = X^7 - 6X^5 + 10X^3 - 4X$$ $$P_8 = X^8 - 7X^6 + 15X^4 - 10X^2 + 1$$
Pour les racines, on a :

```
[0.]
[-1.  1.]
[-1.41421356  0.          1.41421356]
[-1.61803399 -0.61803399  0.61803399  1.61803399]
[-1.73205081 -1.          0.          1.          1.73205081]
[-1.80193774 -1.2469796  -0.44504187  0.44504187  1.2469796   1.80193774]
[-1.84775907 -1.41421356 -0.76536686  0.          0.76536686  1.41421356   1.84775907]
[-1.87938524 -1.53208889 -1.         -0.34729636  0.34729636  1.           1.53208889     1.87938524]
```

On remarque que les racines de $P_n$ sont les valeurs propres de $A_{n,a}$.

### Exercice 4.

```python
import numpy as np
import numpy.linalg as alg
import numpy.random as rd
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.integrate import quad

#On définit le produit scalaire
def prodscal(P, Q):
	def f(t):
		return P(t)*Q(t)

	return quad(f, -1, 1)

E = [Polynomial(1 / np.sqrt(2))]

for k in range(1, 6):
	B = Polynomial( [0]*k + [1] )
	U = B
	for j in range(1, k):
		U = U - prodscal(B, E[j])*E[j]
	invnorme = 1/(np.sqrt(prodscal(U,U)))
	E.append(invnorme*U)

t = np.linspace(-1, 1, 100)
for h in range(len(E)):
	plt.plot(t, E[h](t))

plt.show()
```

