import matplotlib.pyplot as plt
import numpy as np
if __name__ =="__main__":
    u = np.random.uniform(0.0,1.0,10000)
    plt.hist(u, 80, facecolor="r", alpha=0.75)
    plt.grid(True)
    plt.show()

    times =10000
    for time in range(times):
        u += np.random.uniform(0.0,1.0,10000)
        print(len(u))
        u/=times
        print(len(u))
        plt.hist(u, 80, facecolor="g", alpha=0.75)
        plt.grid(True)
        plt.show()