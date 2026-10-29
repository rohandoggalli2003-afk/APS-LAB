from data import generate_nonlinear_data
from visual import plot_2d_data

def main():
    X,y=generate_nonlinear_data()
    plot_2d_data(X,y,title="Non Linearly seperable Data")

if __name__=="__main__":
    main()