#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


df = pd.read_csv(r'C:\Users\Supun\Desktop\Supun Backup\Supun\transfer\documents\myfiles\MScDA\DS\NN\assignments_NN\Assignment-02\iris.txt')
df.head()


# In[22]:


#removing last column of the iris dataset
df = df.iloc[:,:-1]
df.dtypes


# In[24]:


#Min max Scaling the data set to vary between [0,1]
from sklearn.preprocessing import MinMaxScaler
feature_scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = feature_scaler.fit_transform(df)
df_scaled = pd.DataFrame(data=df_scaled)
df_scaled.head()


# In[25]:


# Define parameters for SOM 
som_width = 10
som_length = 10
epochs = 15000
initial_learning_rate=0.01
np.random.seed(8)


# In[26]:


raws,cols = df_scaled.shape
print("row_count=%d column_count=%d" %(raws, cols))


# In[27]:


initial_radius = max(som_width, som_length)/2
time_constant =  epochs/np.log(initial_radius)


# In[28]:


som_net = np.random.random((som_width, som_length, cols))
print("Initial weights set to SOM network:")
print(som_net)


# In[29]:


#define basic functions
def update_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def update_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)

def calculate_euclidian_dis(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# In[30]:


#Logic to calculcate best matching unit
def find_best_matching_Unit(data_point):
    bmu_pos = np.array([0, 0])
    min_dist = np.iinfo(np.int).max
    input_dim = len(data_point)
    
    for x in range(som_width):
        for y in range(som_length):
            som_weight_vector = som_net[x, y, :].reshape(1, 4)
            euclidian_dist = calculate_euclidian_dis(som_weight_vector, data_point)
            if euclidian_dist < min_dist:
                min_dist = euclidian_dist
                bmu_pos = np.array([x, y])
    
    bmu = som_net[bmu_pos[0], bmu_pos[1], :].reshape(1, 4)
    return (bmu, bmu_pos)


# In[31]:


#Neighbourhood function to calculate influence from best matching unit and selected node
def neighbourhood_function(bmu_location, selected_node_location, radius):
    euclidien_dist_to_bmu = calculate_euclidian_dis(bmu_location, selected_node_location)
    return np.exp(-euclidien_dist_to_bmu / (2* (radius**2)))


# In[32]:


#Train SOM network with Iris data set
#shuffle data set
df_scaled = df_scaled.sample(frac=1)

rad_values = list()
learn_rates_values = list()
rad_values.append(initial_radius)
learn_rates_values.append(initial_learning_rate)

for i in range(epochs):
    data_point = np.array(df_scaled.sample())
    bmu, bmu_idx = find_best_matching_Unit(data_point)

    r_new = update_radius(initial_radius, i, time_constant)
    new_learning_rate = update_learning_rate(initial_learning_rate, i, epochs)
    
    rad_values.append(r_new)
    learn_rates_values.append(new_learning_rate)
    
    for x in range(som_width):
        for y in range(som_length):
            w = som_net[x, y, :].reshape(1, 4)
            w_dist = calculate_euclidian_dis(np.array([x, y]), bmu_idx)
            
            if w_dist <= r_new:
                influence = neighbourhood_function(bmu, w, r_new)
                new_w = w + (new_learning_rate * influence * (data_point - w))
                som_net[x, y, :] = new_w.reshape(1, 4)


# In[33]:


from matplotlib import pyplot as plt
plt.plot(rad_values)
plt.title('Radius values')


# In[34]:


plt.plot(learn_rates_values)
plt.title('Learning Rates values')


# In[35]:


#Visualize the weights of the SOM after number of epoch times
from matplotlib import patches as patches

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1, aspect='equal')
ax.set_xlim((0, som_width+1))
ax.set_ylim((0, som_length+1))
ax.set_title('SOM after %d iterations' % epochs)

for x in range(1, som_width + 1):
    for y in range(1, som_length + 1):
        ax.add_patch(patches.Circle((x, y), 0.5, facecolor=som_net[x-1,y-1,:], edgecolor='black'))
plt.show()

fig.savefig('SOM_fig.png')


# In[36]:


#U Matrix Calculation from above SOM
u_matrix = np.zeros((som_width-1, som_length-1))

for x in range(1, som_width):
    for y in range(1, som_length):
        neighbour_list = list()
        print("-"* 100)
        print("neighbour cordinates of x=%d, y=%d" %(x,y))
        for u in range(x-1, x+2):
            if (u < 0 or u > (som_width-1)):
                continue
            for v in range(y-1, y+2):
                if(v < 0 or v > (som_length-1)):
                    continue
                if (u == x and v == y):
                    continue
                neighbour_list.append(np.array([u,v]))
                print(u,v)
        sum=0
        for idx in neighbour_list:
            sum += calculate_euclidian_dis(som_net[x,y,:], som_net[idx[0],idx[1],:])
        
        avg = sum/len(neighbour_list)
        print("Sum of distance to neighbour weights=%f, average=%f" % (sum, avg))     
        u_matrix[x-1,y-1] = avg


# In[37]:


fig = plt.figure(figsize=(6,6))
plt.title("U Matrix visualization using SOM")
plt.imshow(u_matrix, cmap="Reds")
plt.show()
fig.savefig('U_Matrix.png')


# In[ ]:




