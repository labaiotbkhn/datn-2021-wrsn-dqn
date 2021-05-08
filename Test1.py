import matplotlib.pyplot as plt
plt.axis([0, 10, 0, 10])


for i in range(10):
    plt.scatter(i, i + 1)
    plt.pause(0.5)

plt.show()
