import matplotlib.pyplot as plt

def plot_2d_data(X,y,title="Linear Data"):
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
