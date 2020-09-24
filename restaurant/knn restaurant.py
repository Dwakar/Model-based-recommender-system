import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
ratings = pd.read_csv('rating_final.csv')
cuisine= pd.read_csv('usercuisine.csv')
new_df = pd.merge(ratings,cuisine,on='userID').drop(['placeID','food_rating','service_rating'],axis=1) 
missing_pivot = new_df.pivot_table(values='rating',index='userID',columns='Rcuisine')
rate = {}
rows_indexes = {}
for i,row in missing_pivot.iterrows():
	rows = [x for x in range(0,len(missing_pivot.columns))]#find the length of column
	combine = list(zip(row.index, row.values, rows))
	rated = [(x,z) for x,y,z in combine if str(y) !='nan'] #taking only non nan values
	index = [i[1] for i in rated] #indexes of rated cuisine
	row_names = [i[0] for i in rated] #names of rated cuisine
	rows_indexes[i] = index
	rate[i] = row_names
pivot_table = new_df.pivot_table(values= 'rating', index= 'userID', columns ='Rcuisine').fillna(0)
pivot_table = pivot_table.apply(np.sign)
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

n=5 
cosine_nn = NearestNeighbors(n_neighbors = n,algorithm='brute', metric = 'cosine')
item_cosine_nn_fit = cosine_nn.fit(pivot_table.T.values) #transpose the columns users and cuisine
item_distances, item_indices = item_cosine_nn_fit.kneighbors(pivot_table.T.values)

 #for recommender
item_dic = {}
for i in range(len(pivot_table.T.index)):
	item_idx = item_indices[i]
	col_names = pivot_table.T.index[item_idx].tolist()
	item_dic[pivot_table.T.index[i]] = col_names

topRecs = {}
for k,v in rows_indexes.items():
	item_idx = [j for i in item_indices[v] for j in i] #find all the indexes of the cuisine similar to which they have rated
	item_dist= [j for i in item_distances[v] for j in i]#find all the distances  of the cuisine similar to which they have rated
	combine = list(zip(item_dist,item_idx)) #comine distance and idexes found 
	diction = {i:d for d, i in combine if i not in v} #new dictionary with distance and indices except for those already done
	zipped = list(zip(diction.keys(), diction.values()))
	sort = sorted(zipped,key=lambda x:x[1])
	recommendations =[(pivot_table.columns[i],d) for i,d in sort]
	topRecs[k] = recommendations #k -users

def getrecommendations(user,number_of_recs =30):

	print("These are all the cuisine you have viewed view in the past: \n\n{}".format('\n'.join(rate[user])))
	print()
	print("we recommend to view these cuisine too:\n")

	for k,v in topRecs.items():
		if user==k:
			for i in v[:number_of_recs]:
				print('{} with similarity: {:.4f}'.format(i[0],1-i[1]))	

getrecommendations('U1004')#user id

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
