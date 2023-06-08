import numpy as np
from matplotlib import pyplot as plt

sin_x = np.zeros((2001,1000))
sin_x = sin_x-1
aa=[]
cnt_h = []
kk=[]
cc=0

for K in range(2001):
    kk.append(K)

for k in range(0, 4001, 2):

    x = []
    x.append(0.1)
    xx=0.1

    a = k/1000
    a = np.round(a, decimals=3)
    aa.append(a)
    t = []
    n=1000
    N=1000

    for i in range(n):
        #print(i+1)
        xxx=a * xx * (1-xx)
        x.append(xxx)
        xx=xxx
        t.append(i)

    t.append(n)

    if a>=2.95 and a<=3.05:
        x_round = np.round(x, decimals=9)
    else:
        x_round = np.round(x, decimals=5)

    #周期解の検出
    cnt = 1
    c1=x_round[80]
    c2=x_round[81]

    #print(c1)
    #print(c2)

    for i in range(82, 1000):
        if c1==c2:
            break
        cnt += 1

        if c1==x_round[i] and c2==x_round[i+1]:
            break

    
    if a < 1:
        cnt_h.append(0)
    else:
        cnt_h.append(cnt)

    #print(cnt)

    for l in range(cnt):
        if a < 1:
            sin_x[kk[cc],0] = 0
            break
        sin_x[kk[cc],l] = x[80+l]
    
    print(a)
    cc+=1


X = []
XX = []
for i in range(0, 100, 1):
    X.append(a*(i/100)*(1-(i/100)))
    XX.append(i/100)
#print(X)


#図の描画
plt.title("Logistic map bifurcation", {"fontsize":25})
plt.xlabel("a", {"fontsize":20})
plt.ylabel("x(t)", {"fontsize":20})
plt.plot(aa, sin_x, ",r")
plt.legend()
plt.xlim(0, 4)
plt.ylim(0, 1)
plt.show()

