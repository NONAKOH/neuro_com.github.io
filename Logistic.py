#分岐が2以上
import numpy as np
from matplotlib import pyplot as plt

x = []
x.append(0.1)
xx=0.1

a = 4

t = []
n=100


for i in range(n):
    print(i+1)
    xxx=a * xx * (1-xx)
    x.append(xxx)
    xx=xxx
    t.append(i)

t.append(n)
#print(x)
#print(t)

#図の描画
plt.title("Logistic map with a = "+ str(a) +" (4322537)", {"fontsize":25})
plt.xlabel("t", {"fontsize":20})
plt.ylabel("x(t)", {"fontsize":20})
plt.plot(t,x, "-+r")
plt.legend()
plt.xlim(0, n)
plt.ylim(0, 1)
plt.show()

x_round = np.round(x, decimals=4)
#周期解の検出
cnt = 1
c1=x_round[80]
c2=x_round[81]

#print(c1)
#print(c2)

for i in range(82, n):
    if c1==c2:
        break
    cnt += 1

    if c1==x_round[i] and c2==x_round[i+1]:
        break


print(cnt)
y = []
for i in range(cnt):
    y.append(x[80+i])
print(y)

xX=[]
yY=[]
xX.append(y[0])
for i in range(1, cnt):
    xX.append(y[i])
    yY.append(y[i])
    xX.append(y[i])
    yY.append(y[i])
yY.append(y[0])
xX.append(y[0])
yY.append(y[0])
xX.append(y[0])
yY.append(y[1])
print(xX)
print(yY)

X = []
XX = []
for i in range(0, 101, 1):
    X.append(a*(i/100)*(1-(i/100)))
    XX.append(i/100)
#print(X)


plt.title("Logistic map with a = "+ str(a) + ", cycle = "+ str(cnt)+" (4322537)", {"fontsize":25})
plt.xlabel("x(t)", {"fontsize":20})
plt.ylabel("x(t+1)", {"fontsize":20})
plt.plot(XX, X, "-b")
plt.plot(XX, XX, "-g")
plt.plot(xX, yY, "-r")
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
