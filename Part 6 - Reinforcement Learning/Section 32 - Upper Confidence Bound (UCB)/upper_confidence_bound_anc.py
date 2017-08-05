#Upper Confidence Bound

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


#Implementing UCB
import math
N = 10000
d=10
ads_selected = []
number_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0  
for n in range(0,N):
     max_upper_bound =0
     ad=0
     for i in range(0,d):
         if number_of_selections[i]>0:
             average_reward =  sums_of_rewards[i] / number_of_selections[i]
             delta_i = math.sqrt(3/2 * math.log(n + 1)/ number_of_selections[i])
             upper_bound = average_reward + delta_i
         else:
             upper_bound = 1e400
         if upper_bound > max_upper_bound:
             max_upper_bound = upper_bound
             ad = i
         
     ads_selected.append(ad)
     number_of_selections[ad] =  number_of_selections[ad] + 1
     sums_of_rewards[ad] = sums_of_rewards[ad] + dataset.values[n,ad]
     total_reward = total_reward + dataset.values[n,ad]
     
#Visualising the results
plt.hist(ads_selected)
plt.title('Histograms of ad selected')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()     
            
           
                             