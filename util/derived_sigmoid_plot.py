import matplotlib.pyplot as plt
import csv

x = []
y = []
i=0
with open('derived_sigmoid.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[2]))
        y.append(float(row[3]))
        i+=1

plt.plot(x,y, label='8-bit sigmoid function')
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.title('8-Bit sigmoid function')
plt.legend()
plt.show()