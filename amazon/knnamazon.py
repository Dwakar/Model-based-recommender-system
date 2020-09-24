import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv('ratings.csv')
products= pd.read_csv('prodcutamazon.csv',header= 0, encoding= 'unicode_escape')

#creatine new merged dataset
#drop unwanted columns
#axis for which row
new_df = pd.merge(ratings,products,on='productID').drop(['Categories'],axis=1) 

#creating pivot talble for products and users according to their ratings
missing_pivot = new_df.pivot_table(values='ratings',index='username',columns='name')

#Identifying the products that users have rated
rate = {}
rows_indexes = {}
for i,row in missing_pivot.iterrows():
	rows = [x for x in range(0,len(missing_pivot.columns))]#find the length of column
	combine = list(zip(row.index, row.values, rows))
	rated = [(x,z) for x,y,z in combine if str(y) !='nan'] #taking only non nan values
	index = [i[1] for i in rated] #indexes of rated products
	row_names = [i[0] for i in rated] #names of rated products
	rows_indexes[i] = index
	rate[i] = row_names

#repalcing all the nan values with the 0 in the pivot table
pivot_table = new_df.pivot_table(values= 'ratings', index= 'username', columns ='name').fillna(0)

#to help with outliers convert into binary 1 and 0 since some products may not been seen by many user
pivot_table = pivot_table.apply(np.sign)

#for not rated products
notrated = {}
notrated_indexes = {}
for i, row in pivot_table.iterrows():
	rows = [x for x in range(0,len(missing_pivot.columns))]
	combine = list(zip(row.index,row.values, row))
	idx_row = [(idx,col) for idx, val, col in combine if not val>0]
	indices = [i[1] for i in idx_row]
	row_names = [i[0] for i in idx_row]
	notrated_indexes[i] =  indices
	notrated[i] = row_names

#NN Recommender algorithm
n=7
cosine_nn = NearestNeighbors(n_neighbors = n,algorithm='brute', metric = 'cosine')
item_cosine_nn_fit = cosine_nn.fit(pivot_table.T.values) #transpose the columns users and products
item_distances, item_indices = item_cosine_nn_fit.kneighbors(pivot_table.T.values)

 #for recommender
item_dic = {}
for i in range(len(pivot_table.T.index)):
	item_idx = item_indices[i]
	col_names = pivot_table.T.index[item_idx].tolist()
	item_dic[pivot_table.T.index[i]] = col_names

topRecs = {}
for k,v in rows_indexes.items():
	item_idx = [j for i in item_indices[v] for j in i] #find all the indexes of the products similar to which they have rated
	item_dist= [j for i in item_distances[v] for j in i]#find all the distances  of the products similar to which they have rated
	combine = list(zip(item_dist,item_idx)) #comine distance and idexes found 
	diction = {i:d for d, i in combine if i not in v} #new dictionary with distance and indices except for those already done
	zipped = list(zip(diction.keys(), diction.values()))
	sort = sorted(zipped,key=lambda x:x[1])
	recommendations =[(pivot_table.columns[i],d) for i,d in sort]
	topRecs[k] = recommendations #k -users

def getrecommendations(user,number_of_recs =30):


	print("These are all the products you have viewed view in the past: \n\n{}".format('\n'.join(rate[user])))
	print()
	print("we recommend to view these products too:\n")

	for k,v in topRecs.items():
		if user==k:
			for i in v[:number_of_recs]:
				print('{} with similarity: {:.4f}'.format(i[0],1-i[1]))	

getrecommendations('BySharon Lambert')#user id


item_distances = 1-item_distances #for similarity score
predictions = item_distances.T.dot(pivot_table.T.values)/np.array([np.abs(item_distances.T).sum(axis=1)]).T
ground_truth = pivot_table.T.values[item_distances.argsort()[0]]

def rmse(predictions,ground_truth):
	predictions = predictions[ground_truth.nonzero()].flatten()
	ground_truth = ground_truth[ground_truth.nonzero()].flatten()
	return sqrt(mean_squared_error(predictions,ground_truth))

error_rate = rmse(predictions,ground_truth)
print("Accuracy: {:.3f}".format(100-error_rate))
print("RMSE: {:.5f}".format(error_rate))
print("MAE:" + str(mean_absolute_error(predictions, ground_truth)))
