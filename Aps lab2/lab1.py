import matplotlib.pyplot as plt

def check_probability(n):
    prob=1.0
    for i in range(n):
        prob = prob*((365-i)/365)
    return prob
res =[]
for i in range(1, 101):
    res.append(1- check_probability(i))
    print(f'probablility of person {i} is ', 1- check_probability(i))


x_axis = range(1,101)
y_axis =  res

plt.scatter(x_axis, y_axis)
plt.show()