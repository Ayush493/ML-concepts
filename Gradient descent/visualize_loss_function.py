import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


data = pd.read_csv("/Users/ayush.bhatnagar/Work/extra/ML-concepts/Data_sets/generated_3d_data_linear_regression.csv")
print(data)

fig = plt.figure()

'''
#Visualize the data
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.scatter(data.iloc[:,1], data.iloc[:,2], data.iloc[:,0], c='r', marker='o' )
plt.show()
'''

w1_range = range(-100,151)
bias=2

'''
#visualize the loss function by varying only one parameter i.e. w1
result=[]
w2=2
for w1 in w1_range:
    sum=0
    for i in range(0,data.shape[0]):
        y = data.iloc[i,0]
        x1 = data.iloc[i,1]
        x2 = data.iloc[i,2]
        temp = pow((y - w1*x1 - w2*x2 - bias), 2)
        sum+=temp
    result.append(sum)

#plotting value of loss_function for various values of w1
ax = fig.add_subplot(121,projection='3d')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Sum_of_Loss_function')
ax.scatter(w1_range, w2*np.ones(len(result)), result, c='r', marker='o' )
#Did u see the parabola.... :)
'''


#Now let's vary value of w2 as well.
w2_range = range(-100,151)
w1_plot_range=[]
w2_plot_range=[]
result=[]
surface_result = np.zeros((len(w2_range),len(w1_range))) #making a 2d array that will save values at each point of w1 and w2.
for i,w1 in zip(range(len(w1_range)),w1_range):
    for j,w2 in zip(range(len(w2_range)),w2_range):
        sum=0
        for k in range(0,data.shape[0]):
            y = data.iloc[k,0]
            x1 = data.iloc[k,1]
            x2 = data.iloc[k,2]
            temp = pow((y - w1*x1 - w2*x2 - bias), 2)
            sum+=temp
        result.append(sum)
        w1_plot_range.append(w1)
        w2_plot_range.append(w2)
        surface_result[j][i]=sum

'''
#plotting value of loss_function for various values of w1 and w2
ax = fig.add_subplot(121,projection='3d')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Sum_of_Loss_function')
ax.scatter(w1_plot_range, w2_plot_range, result, c='r', marker='o' )
#We got a cup shape but we are unable to visualize it properly so let's plot things in a more better way.
'''

w1_surf_plot_range,w2_surf_plot_range = np.meshgrid(w1_range, w2_range) #this will create equal size matrices of [len_w2 x len_w1] for plotting like w1 varying keeping w2 constant. try meshgrid on m=[1,2] and n=[3,4,5].
# So now we have result for each exact point in w1_surf_plot_range and w2_surf_plot_range saved in surface_result. i.e.
# for combination of w1 value at w1_surf_plot_range[2][3] and w2 value at w2_surf_plot_range[2][3] we have loss function value stored at surface_result[2][3].
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Sum_of_Loss_function')
surf = ax.plot_surface(w1_surf_plot_range, w2_surf_plot_range, surface_result, cmap=cm.coolwarm)

plt.show()

#Almost a bowl :p