import pyadlahelper
import pandas as pd
import numpy as np
import sklearn.cluster as cluster

def ward_clustering(input_doc_feat, input_doc_weight,n_clusters=100):
	model = cluster.AgglomerativeClustering(n_clusters=n_clusters) # 3.6 -> 2.0 reset number of clusters
	if len(input_doc_feat) < 2:
		cluster_data = input_doc_feat
		cluster_weight = input_doc_weight
		cluster_label = np.zeros(1)
		return cluster_data, cluster_weight, cluster_label
	cluster_label = model.fit_predict(input_doc_feat)
	n_cluster = max(cluster_label) + 1
	cluster_data = np.zeros((n_cluster, input_doc_feat.shape[1]), dtype=np.float32)
	cluster_weight = np.zeros(n_cluster, dtype=np.float32)
	for cluster_id in range(n_cluster):
		index = np.where(cluster_label == cluster_id)
		cluster_data[cluster_id] = np.mean(input_doc_feat[index], axis=0)
		cluster_weight[cluster_id] = np.sum(input_doc_weight[index])
	return cluster_data, cluster_weight, cluster_label

'''
def kmeans_clustering(input_doc_feat, input_doc_weight):
	model = cluster.KMeans(n_clusters = 2000);
	input_doc_weights = input_doc_weight
	if len(input_doc_feat) < 2:
		cluster_data = input_doc_feat
		cluster_weight = input_doc_weight
		cluster_label = np.zeros(1)
		return cluster_data, cluster_weight, cluster_label
	cluster_label = model.fit_predict(input_doc_feat)
	n_cluster = max(cluster_label) + 1
	cluster_data = np.zeros((n_cluster, input_doc_feat.shape[1]), dtype=np.float32)
	cluster_weight = np.zeros(n_cluster, dtype=np.float32)
	for cluster_id in range(n_cluster):
		index = np.where(cluster_label == cluster_id)
		cluster_data[cluster_id] = np.mean(input_doc_feat[index], axis=0)
		cluster_weight[cluster_id] = np.sum(input_doc_weights[index])
	return cluster_data, cluster_weight, cluster_label
'''

class WardClusterReducer:
	def __init__(self, in_n_clusters):
		self.in_n_clusters = int(in_n_clusters)
		#self.delim = delim
		
	def Reduce(self, inputRowset, outputRow):
		input_dataframe = pyadlahelper.RowsetToDataframe(inputRowset, -1)
		input_doc_feat = np.array(input_dataframe.ImgEmbedding.str.split(',').tolist(), dtype=np.float)
		input_doc_weight = input_dataframe.loc[:, 'Weight'].values
		input_doc_weight = np.array(list(input_doc_weight), dtype=np.float)
		cluster_feat, cluster_weight, cluster_label = ward_clustering(input_doc_feat, input_doc_weight, n_clusters = self.in_n_clusters)
		max_cluster_labels = len(cluster_label)
		for i in range(len(input_doc_feat)):
			cluster_index = int(cluster_label[i])
			outputRow['Key'] = input_dataframe.loc[i,'Key']
			# outputRow['ClusterEmbedding'] = (' '.join(list(map(str, cluster_feat[i]))))
			outputRow['ClusterWeight'] = cluster_weight[cluster_index].astype(float)
			outputRow['ClusterIndex'] = cluster_index
			yield outputRow
