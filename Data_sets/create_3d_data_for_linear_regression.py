import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D #this line is needed for 3D plotting
import matplotlib.pyplot as plt

# y = w1 * x1 + w2 * x2 + w0

# let me choose my own values of parameters
w0 = 5
w1 = 50
w2 = 10

# Generating 10 random values of x1 and x2:
np.random.seed(3)
x1 = np.random.randint(10, size=10)
x2 = np.random.randint(10, size=10)
y = w1 * x1 + w2 * x2 + w0

print(x1)
print(x2)
print(y)

# Let's visualize it:
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.scatter(x1, x2, y, c='r', marker='o' )
plt.show()

#Save it: 1st we will create a pandas dataframe and then save it to csv
df = pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})
df.to_csv("/Users/ayush.bhatnagar/Work/extra/ML-concepts/Data_sets/generated_3d_data_linear_regression.csv",columns=["y","x1","x2"],index=False) #columns argument for preserving the order and index argument for not writing row numbers.
