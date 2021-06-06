import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


x = [0, 15, 30, 50, 90, 150]
y = [0.09486, 0.71181, 0.93012, 1, 0.85616, 0.74518]


# b = plt.scatter(x, y)
# plt.show()
z1 = np.polyfit(x, y, 2)  # 用1次多项式拟合
p1 = np.poly1d(z1)
print('fitting func: \n', p1)  # 在屏幕上打印拟合多项式

yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
plot1 = plt.plot(x, y, '*', label='original values')
plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc=4)
plt.title('polyfitting')
plt.show()
