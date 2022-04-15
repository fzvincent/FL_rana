import numpy as np
from matplotlib import pyplot as plt
from var import *
sizeH=9
sizeV=2.5
mydata=np.load("mydata0406.npz")
validData = mydata["validData"]

# 0 for FedAvg
# 1 for Green
# 2 for Learn
a=np.average(validData,axis=3)


for i in range(betaCount):
    plt.plot(VALID_RANGE, validData[0, i, :, 0], label='FedAvg')
    plt.plot(VALID_RANGE, validData[1, i, :, 0], label='Green')
    plt.plot(VALID_RANGE, validData[2, i, :, 0], label='Learn')
    plt.grid()
    plt.legend()
    plt.xlabel('Global iterations')
    plt.ylabel('Test accuracy')

    plt.savefig('f' + str(i) + '.pdf')
    plt.show()







