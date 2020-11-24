import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
import matplotlib
from cycler import cycler
import pandas as pd
import csv
from pandas_profiling import ProfileReport
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon
from matplotlib import colors

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold, LeaveOneOut

from sklearn.preprocessing import Normalizer
import random

from matplotlib.font_manager import FontProperties

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# https://github.com/scikit-learn-contrib/imbalanced-learn/tree/0eb90333ca3b0095b430ac3976ffe1b3e6fe9be5/imblearn/over_sampling
from imblearn.over_sampling import SMOTE

from sklearn.pipeline import make_pipeline

from sklearn.metrics import silhouette_score

#
#
#
#
#
#
# PARKINSONS CLASS
# CHIAMARE METODO .start() lanciare il programma
#
#
#
#
#
#
class Parkinsons:
	
	# Constructor
	def __init__(self):

		self.RANDOM_STATE = 42

		return

	# Methods

	def smote_simulation(self, X_train, y_train):

		np.random.seed(82)

		scaler = StandardScaler(copy=True)
		scaler.fit(X_train)

		X_train = scaler.transform(X_train)
		# X_test = scaler.transform(X_test)

		pca = PCA(n_components=2)
		pca.fit(X_train)
		X_train = pca.transform(X_train)

		X_train = np.transpose(X_train)

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))

		ax.scatter(X_train[0,y_train==0], X_train[1,y_train==0], c='blue', alpha=0.7, marker='o')
		ax.scatter(X_train[0,y_train==1], X_train[1,y_train==1], c='red', alpha=0.7, marker='^')

		# plt.show()

		# sys.exit()

		print('D1', list(X_train[:,y_train==1]))
		print('D2', list(X_train[:,y_train==0]))

		D2_saved = np.array(X_train[:,y_train==0])

		# print(D2_saved.shape)

		# sys.exit()

		# X = np.concatenate((np.transpose(D1), np.transpose(D2)))
		# y = np.concatenate((np.zeros(n1), np.ones(n2)))

		# print(len(y))
		# print(len(X))

		X_train = np.transpose(X_train)

		sampler = SMOTE()
		X_train, y_train = sampler.fit_resample(X_train, y_train)

		# print(len(X))
		# print(len(y))
		# unique, counts = np.unique(y, return_counts=True)
		# print(np.asarray((unique, counts)).transpose())

		D2_new = X_train[y_train==0,:]

		D2_new = {(point[0], point[1]) for point in D2_new}
		D2_old = {(point[0], point[1]) for point in np.transpose(D2_saved)}

		# print(len(D2_new))
		# print(len(D2_old))

		D2_syntetic = D2_new - D2_old # Prendo solo quelli nuovi generati

		# print(len(D2_syntetic))

		D2_syntetic = np.transpose(np.array([[point[0], point[1]] for point in D2_syntetic]))

		# print(D2_syntetic.shape)

		ax.scatter(D2_syntetic[0,:], D2_syntetic[1,:], c='black', alpha=0.7, marker='s')

		print('D2 syntetic', list(D2_syntetic))

		plt.show()

		return
	
	def start(self):

		# self.final_confmat()
		# sys.exit()

		# self.final_graphs()
		# sys.exit()

		X, y, X_individuals, y_individuals, names_individuals, columns, names = self.get_dataset()

		X = np.array(X, dtype=np.float64)
		y = np.array(y, dtype=np.int32)

		X_individuals = np.array(X_individuals, dtype=np.float64)
		y_individuals = np.array(y_individuals, dtype=np.int32)

		# print('TOTAL ORIGINAL DATASET:', X.shape)
		# unique, counts = np.unique(y, return_counts=True)
		# print('Original class distribution\n', np.asarray((unique, counts)).transpose())

		# print('TOTAL INDIVIDUALS DATASET:', X_individuals.shape)
		# unique, counts = np.unique(y_individuals, return_counts=True)
		# print('Individuals class distribution\n', np.asarray((unique, counts)).transpose())

		# ----- EXPLORATORY DATA ANALYSIS
		# self.dataExploration(X, y, X_individuals, y_individuals, columns, names, names_individuals)
		# sys.exit()
		# -------------------------------

		(X_train,
		X_test,
		y_train,
		y_test, y_test_patients,
		X_train_individuals,
		X_test_individuals,
		y_train_individuals,
		y_test_individuals) = self.custom_train_test_split(X, y, X_individuals, y_individuals,
														   names_individuals, columns, names,
														   test_size=0.25, stratify=True)

		# unique_names = {name for name in y_test_patients}
		
		# for name in unique_names:
		# 	print((y_test_patients==name) & (y_test == 1))
		# sys.exit()
		# print(y_test)
		# print(y_test_patients)

		# --- SMOTE SIMULATION
		# self.smote_simulation(X_train, y_train)

		print('\n----------------------\n')

		# df_X = pd.DataFrame(X)
		# print(df_X.describe())

		# ---------- DATA PREPROCESSING

		# ---- Down to 12 features
		X_train, X_test, columns_new = self.remove_linear_correlated_features(X_train, X_test, columns, display=True)
		# ------------------------

		# print(X_train)
		# print(columns_new)

		scaler = StandardScaler(copy=True)
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)

		# scaler = MinMaxScaler(copy=True)
		# scaler.fit(X_train)
		# X_train = scaler.transform(X_train)
		# X_test = scaler.transform(X_test)

		# --- PCA 3D scatter plot
		# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4.5))
		# ax = Axes3D(fig)
		# ax.scatter(X_train[y_train==0,0], X_train[y_train==0,1], X_train[y_train==0,2], s=10, c='blue', alpha=0.8, marker='o')
		# ax.scatter(X_train[y_train==1,0], X_train[y_train==1,1], X_train[y_train==1,2], s=10, c='red', alpha=0.8, marker='o')
		# plt.show()
		# sys.exit()
		# ----------------------

		# ----- PCA graph
		# self.pca_cumulative_graph(pca)
		# ---------------

		# --- GRID-SEARCH WITH LOOV CROSS-VALIDATION

		self.DO_PCA = True
		self.DO_SMOTE = True
		self.DO_THIS_CLASSIFIER = 'LogReg'  # DecisionTree, RandomForest, svm, rbfsvm,  LogReg, knn

		if self.DO_PCA:
			pca = PCA(n_components=5)
			pca.fit(X_train)
			X_train = pca.transform(X_train)
			X_test = pca.transform(X_test)

		classifier_params = {
						'DecisionTree': {'name': 'DecisionTree', 'classifier': DecisionTreeClassifier},
						'RandomForest': {'name': 'RandomForest', 'classifier': RandomForestClassifier},
						'svm': {'name': 'svm', 'classifier': SVC},
						'rbfsvm': {'name': 'rbfsvm', 'classifier': SVC},
						'LogReg': {'name': 'LogReg', 'classifier': LogisticRegression},
						'knn' : {'name': 'knn', 'classifier': KNeighborsClassifier}
		}

		params = {
			'DecisionTree': {'max_depth': [None, 4, 8],
							 'splitter': ['best'],
							 'min_impurity_decrease': [0.0, 0.02, 0.05],
							 'min_samples_split': [2, 4],
							 'random_state': [74]
							},
			# 'RandomForest': {
			# 			     'n_estimators': [1,5,10,20,30,40,50,60,80,100]
			# 				},
			'RandomForest': {'max_depth': [None, 4, 8],
						     'min_samples_split': [2, 4],
						     'min_impurity_decrease': [0.0, 0.02, 0.05],
						     'n_estimators': [100]
							},
			'svm': {
						'C': [0.01, 0.1, 1.0, 10.0, 100.0],
						'kernel': ['linear']
					},
			'rbfsvm': {
						'C': [0.01, 0.1, 1.0, 10.0, 100.0],
						'gamma': ['scale'],
						'kernel': ['rbf']
					},
			'LogReg': {
						'C': [0.01, 0.1, 1.0, 10.0, 100.0]
					},
			'knn': {
					'n_neighbors': [11]
			}
		}
		
		if not self.DO_SMOTE:
			self.grid_search_loocv(X_train, y_train, X_test, y_test, y_test_patients,
								params[classifier_params[self.DO_THIS_CLASSIFIER]['name']], classifier_params[self.DO_THIS_CLASSIFIER]['classifier'], pos_label=1, average='macro', columns_new=columns_new)
		else:
			self.grid_search_loocv_SMOTE(X_train, y_train, X_test, y_test, y_test_patients,
								params[classifier_params[self.DO_THIS_CLASSIFIER]['name']], classifier_params[self.DO_THIS_CLASSIFIER]['classifier'], pos_label=1, average='macro')

		sys.exit()

		# --- VALUTAZIONI OUT-OF-THE-BOX

		clf0 = DecisionTreeClassifier(max_depth=None, splitter='best', min_impurity_decrease=0.0, min_samples_split=4)
		clf1 = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, max_features='auto', min_impurity_decrease=0.0)
		clf2 = SVC(C=10.0, kernel='rbf', gamma='scale', class_weight=None)
		clf3 = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True)
		clf5 = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')

		clf0.fit(X_train, y_train)
		clf1.fit(X_train, y_train)
		clf2.fit(X_train, y_train)
		clf3.fit(X_train, y_train)
		y_pred0 = clf0.predict(X_test)
		y_pred1 = clf1.predict(X_test)
		y_pred2 = clf2.predict(X_test)
		y_pred3 = clf3.predict(X_test)

		accuracy0 = accuracy_score(y_test, y_pred0)
		accuracy1 = accuracy_score(y_test, y_pred1)
		accuracy2 = accuracy_score(y_test, y_pred2)
		accuracy3 = accuracy_score(y_test, y_pred3)

		print('Accuracy clf0:', accuracy0)
		print('Accuracy clf1:', accuracy1)
		print('Accuracy clf2:', accuracy2)
		print('Accuracy clf3:', accuracy3)
		print('Average accuracy: ', (accuracy1+accuracy2+accuracy3+accuracy0)/4)

		p, r, f1, s = precision_recall_fscore_support(y_test, y_pred0, average=None)
		print('clf0 f1_macro:', f1)
		p, r, f1, s = precision_recall_fscore_support(y_test, y_pred1, average=None)
		print('clf1 f1_macro:', f1)
		p, r, f1, s = precision_recall_fscore_support(y_test, y_pred2, average=None)
		print('clf2 f1_macro:', f1)
		p, r, f1, s = precision_recall_fscore_support(y_test, y_pred3, average=None)
		print('clf3 f1_macro:', f1)

		f1 = f1_score(y_test, y_pred0, pos_label=1, average='binary')
		print('clf0 binary', f1)
		f1 = f1_score(y_test, y_pred1, pos_label=1, average='binary')
		print('clf1 binary', f1)
		f1 = f1_score(y_test, y_pred2, pos_label=1, average='binary')
		print('clf2 binary', f1)
		f1 = f1_score(y_test, y_pred3, pos_label=1, average='binary')
		print('clf3 binary', f1)

		f1 = f1_score(y_test, y_pred0, pos_label=1, average='weighted')
		print('clf0 weighted', f1)
		f1 = f1_score(y_test, y_pred1, pos_label=1, average='weighted')
		print('clf1 weighted', f1)
		f1 = f1_score(y_test, y_pred2, pos_label=1, average='weighted')
		print('clf2 weighted', f1)
		f1 = f1_score(y_test, y_pred3, pos_label=1, average='weighted')
		print('clf3 weighted', f1)

		print('\n-------------------------\n')

		# print(sorted(clf1.feature_importances_, reverse=True))
		# print(np.cumsum(sorted(clf1.feature_importances_, reverse=True)))

		# print('Random Forest:', clf1.feature_importances_)
		# print('DecTree:', clf0.feature_importances_)

		# print(export_graphviz(clf0, feature_names=columns_new))

		# ----- Feature importances graph
		# feat_importances = pd.DataFrame({'Feature importance ratio': clf1.feature_importances_, 'names': columns_new})
		# feat_importances.sort_values(by=['Feature importance ratio'], inplace=True, ascending=False)

		# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4.5))
		# ax = sns.barplot("Feature importance ratio", "names", data=feat_importances,
  		#                palette="Blues_d")
		# plt.show()
		# ---------------

		sys.exit()

		return

	def grid_search_loocv_SMOTE(self, X_train, y_train, X_test, y_test, y_test_patients, params, Classifier, oversampling=False, pos_label=1, average='macro'):

		# kf = KFold(n)
		loo = LeaveOneOut()

		# y_pred = list()
		print('---')

		best_f1_score = 0.0
		best_config = None
		
		for configuration in ParameterGrid(params):
			# myFunction(**configuration)	

			# print('SMOTE config:',configuration)

			clf = Classifier(**configuration)
			y_pred = list()

			# Leave one out
			for train_indices, test_indices in loo.split(X_train):
				X_train_curr = X_train[train_indices]
				X_test_curr = X_train[test_indices]
				y_train_curr = y_train[train_indices]
				y_test_curr = y_train[test_indices]
			
				## print('--- SMOTE ---')
				# print(X_train_curr.shape, y_train_curr.shape)
				# unique, counts = np.unique(y_train_curr, return_counts=True)
				# print(np.asarray((unique, counts)).transpose())
				sampler = SMOTE(random_state=42)
				X_train_curr, y_train_curr = sampler.fit_resample(X_train_curr, y_train_curr)
				# print(X_train_curr.shape, y_train_curr.shape)
				# unique, counts = np.unique(y_train_curr, return_counts=True)
				# print(np.asarray((unique, counts)).transpose())
				## print('------')

				clf.fit(X_train_curr, y_train_curr)
				y_pred_curr = clf.predict(X_test_curr)
				y_pred.append(y_pred_curr)

			
			y_pred = np.array(y_pred)
			f1 = f1_score(y_train, y_pred, pos_label=pos_label, average=average)
			# print('clf f1-weighted', f1)
			if f1 > best_f1_score:
				best_f1_score = f1
				best_config = configuration

		print('\nBest configuration found:', best_config)
		print('With f1-score '+str(average)+':', best_f1_score)
		clf = Classifier(**best_config)

		sampler = SMOTE(random_state=42)
		X_train, y_train = sampler.fit_resample(X_train, y_train)

		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)

		accuracy = accuracy_score(y_test, y_pred)
		f1_weighted = f1_score(y_test, y_pred, pos_label=pos_label, average='weighted')


		p_macro, r_macro, f1_macro, s = precision_recall_fscore_support(y_test, y_pred, average='macro')


		p_none, r_none, f1_none, s_none = precision_recall_fscore_support(y_test, y_pred, average=None)
		# print('NUOVE:',p,r,f1)

		conf_mat = confusion_matrix(y_test, y_pred)
		print('CONF\n', conf_mat)
		print('\n')

		print('\nEVALUATION ON TEST SET:')
		print('f1-score (weighted) '+str(average)+':', f1_weighted)
		print('accuracy:', accuracy)

		# --- Compute # of PD patients correctly recalled
		n_pd_patients_recalled = 0
		n_pd_patients = 0
		n_healthy_patients_recalled = 0
		n_healthy_patients = 0
		unique_names = {name for name in y_test_patients}
		
		for name in unique_names:
			if len(y_pred[(y_test_patients==name) & (y_test == 1)]) > 0:
				curr_mean = np.mean(y_pred[(y_test_patients==name) & (y_test == 1)])
				print('PD patient:', curr_mean)
				if curr_mean > 0.5:
					n_pd_patients_recalled += 1
				n_pd_patients += 1
			else:
				curr_mean = 1-np.mean(y_pred[(y_test_patients==name) & (y_test == 0)])
				print('Healty patient:', curr_mean)
				if curr_mean > 0.5:
					n_healthy_patients_recalled += 1
				n_healthy_patients += 1

		print('# PD patients correctly recalled:', str(n_pd_patients_recalled)+'/'+str(n_pd_patients))
		print('# Healthy patients correctly recalled:', str(n_healthy_patients_recalled)+'/'+str(n_healthy_patients))


		print('\n----\n----\n')

		print('CLASSIFIER', self.DO_THIS_CLASSIFIER)
		print('PCA', self.DO_PCA)
		print('SMOTE', self.DO_SMOTE)

		print('accuracy','f1_weighted','f1_macro','precision_macro', 'recall_macro', '#PDPatientsRecalled', '#HealthyPatientsRecalled, p1, r1, p0, r0, f1, f0')
		print([round(accuracy,4), round(f1_weighted,4), round(f1_macro,4), round(p_macro,4), round(r_macro,4), n_pd_patients_recalled, n_healthy_patients_recalled, round(p_none[1],4), round(r_none[1],4), round(p_none[0],4), round(r_none[0],4), round(f1_none[1],4), round(f1_none[0],2)])

		print('\n----\n----\n')

		print('\n--------\nEND GRID-SEARCH\n--------\n')	

		return

	def grid_search_loocv(self, X_train, y_train, X_test, y_test, y_test_patients, params, Classifier, oversampling=False, pos_label=1, average='macro', columns_new=[]):

		best_f1_score = 0.0
		best_config = None

		all_scores = []
		
		for configuration in ParameterGrid(params):
			print(configuration)

			clf = Classifier(**configuration)

			y_pred_val = cross_val_predict(clf, X_train, y_train, cv=LeaveOneOut())
			
			f1 = f1_score(y_train, y_pred_val, pos_label=pos_label, average=average)
			if f1 > best_f1_score:
				best_f1_score = f1
				best_config = configuration
			all_scores.append(f1)
		
		print('all scores:', all_scores)

		print('\nBest configuration found:', best_config)
		print('With f1-score '+str(average)+':', best_f1_score)
		clf = Classifier(**best_config)
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)

		accuracy = accuracy_score(y_test, y_pred)
		f1_weighted = f1_score(y_test, y_pred, pos_label=pos_label, average='weighted')


		p_macro, r_macro, f1_macro, s = precision_recall_fscore_support(y_test, y_pred, average='macro')


		p_none, r_none, f1_none, s_none = precision_recall_fscore_support(y_test, y_pred, average=None)
		# print('NUOVE:',p,r,f1)

		conf_mat = confusion_matrix(y_test, y_pred)
		print('CONF\n', conf_mat)
		print('\n')

		print('\nEVALUATION ON TEST SET:')
		print('f1-score (weighted) '+str(average)+':', f1_weighted)
		print('accuracy:', accuracy)

		# --- Compute # of PD patients correctly recalled
		n_pd_patients_recalled = 0
		n_pd_patients = 0
		n_healthy_patients_recalled = 0
		n_healthy_patients = 0
		unique_names = {name for name in y_test_patients}
		
		for name in unique_names:
			if len(y_pred[(y_test_patients==name) & (y_test == 1)]) > 0:
				curr_mean = np.mean(y_pred[(y_test_patients==name) & (y_test == 1)])
				print('PD patient:', curr_mean)
				if curr_mean > 0.5:
					n_pd_patients_recalled += 1
				n_pd_patients += 1
			else:
				curr_mean = 1-np.mean(y_pred[(y_test_patients==name) & (y_test == 0)])
				print('Healty patient:', curr_mean)
				if curr_mean > 0.5:
					n_healthy_patients_recalled += 1
				n_healthy_patients += 1

		print('# PD patients correctly recalled:', str(n_pd_patients_recalled)+'/'+str(n_pd_patients))
		print('# Healthy patients correctly recalled:', str(n_healthy_patients_recalled)+'/'+str(n_healthy_patients))


		print('\n----\n----\n')

		print('CLASSIFIER', self.DO_THIS_CLASSIFIER)
		print('PCA', self.DO_PCA)
		print('SMOTE', self.DO_SMOTE)

		print('accuracy','f1_weighted','f1_macro','precision_macro', 'recall_macro', '#PDPatientsRecalled', '#HealthyPatientsRecalled, p1, r1, p0, r0, f1, f0')
		print([round(accuracy,4), round(f1_weighted,4), round(f1_macro,4), round(p_macro,4), round(r_macro,4), n_pd_patients_recalled, n_healthy_patients_recalled, round(p_none[1],4), round(r_none[1],4), round(p_none[0],4), round(r_none[0],4), round(f1_none[1],4), round(f1_none[0],2)])

		print('\n----\n----\n')

		# coeff_values = pd.DataFrame({'Coefficient value': clf.coef_[0], 'Features': columns_new})
		# coeff_values.sort_values(by=['Coefficient value'], inplace=True, ascending=False)

		# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4.5))
		# ax = sns.barplot("Coefficient value", "Features", data=coeff_values,
  		# 		               palette="Blues_d")
		# plt.show()

		print('\n--------\nEND GRID-SEARCH\n--------\n')

		#print(export_graphviz(clf, feature_names=columns_new, out_file='tree.dot'))
		# dot_data = export_graphviz(clf, feature_names=columns_new)
		# graph = graphviz.Source(dot_data) 
		# graph.render("ultimo_tree") 

		return

	def final_graphs(self):

		names = [
					'DT', 'DT+OS', 'DT+PCA5', 'DT+PCA5+OS',
					'RF', 'RF+OS', 'RF+PCA5', 'RF+PCA5+OS',
					'SVM', 'SVM+OS', 'SVM+PCA5', 'SVM+PCA5+OS',
					'RBFSVM', 'RBFSVM+OS', 'RBFSVM+PCA5', 'RBFSVM+PCA5+OS',
					'LG', 'LG+OS', 'LG+PCA5', 'LG+PCA5+OS'
				]

		# accuracies = [0.8367,0.8571,0.8571,0.8571,0.8776,0.8776,0.9184,0.9184,0.8163,0.898,0.8776,0.898,0.8571,0.8571,0.8776,0.898,0.8776,0.898,0.8776,0.9388]
		# f1_macro = [0.7012,0.7509,0.7509,0.7509,0.7958,0.7958,0.8744,0.8744,0.6797,0.8368,0.7958,0.8368,0.7509,0.7509,0.7958,0.8368,0.7958,0.8368,0.7958,0.9148]
		# p_macro = [0.9111,0.9205,0.9205,0.9205,0.9302,0.9302,0.9512,0.9512,0.8091,0.9405,0.9302,0.9405,0.9205,0.9205,0.9302,0.9405,0.9302,0.9405,0.9302,0.9282]
		# r_macro = [0.6667,0.7083,0.7083,0.7083,0.75,0.75,0.8333,0.8333,0.6532,0.7917,0.75,0.7917,0.7083,0.7083,0.75,0.7917,0.75,0.7917,0.75,0.9032]
		# f1_1 = [0.9024,0.9136,0.9136,0.9136,0.925,0.925,0.9487,0.9487,0.8889,0.9367,0.925,0.9367,0.9136,0.9136,0.925,0.9367,0.925,0.9367,0.925,0.96]
		f1_0 = [0.5,0.59,0.59,0.59,0.67,0.67,0.8,0.8,0.47,0.74,0.67,0.74,0.59,0.59,0.67,0.74,0.67,0.74,0.67,0.87]

		name = 'F1-score (0)'

		values = pd.DataFrame({name: f1_0, 'Features': names})
		values.sort_values(by=[name], inplace=True, ascending=False)

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
		ax = sns.barplot(name, "Features", data=values,
  				               palette="Reds_d")

		plt.xticks(np.arange(0.0, 1.0+0.1, 0.1))
		plt.grid(alpha=0.25, axis='x', linestyle='--')

		fig.savefig('final_'+name+'.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
		# plt.show()

		return

	def final_confmat(self):

		font = FontProperties()
		font.set_family(['Times New Roman', 'serif'])
		font.set_size(9)

		# TIPO = 'LOGREG'

		# conf_mat = np.array([[8,4], [0,37]])
		# conf_mat = np.array([[4,8],[0,37]]) #DT
		# conf_mat = np.array([[8,4],[0,37]]) #RF+PCA+OS
		# conf_mat = np.array([[7,5],[0,37]]) #SVM+PCA5+OS
		# conf_mat = np.array([[10,2],[1,36]]) #LOGREG

		conf_mat = np.array([[2,0],[0,6]])
		# conf_mat = np.array()
		# conf_mat = np.array()
		# conf_mat = np.array()
		# conf_mat = np.array()

		fig, ax = plt.subplots(nrows=1, ncols=1)

		im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.GnBu)
		# ax.figure.colorbar(im, ax=ax)

		ax.set(yticks=[0, 1], 
		       xticks=[0, 1], 
		       yticklabels=['Healthy', 'PD'],
		       xticklabels=['Healthy', 'PD'])
		ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))

		for i in range(2):
		    for j in range(2):
		        text = ax.text(j, i, conf_mat[i, j], ha="center", va="center", color="#00000090", fontweight='bold', fontproperties=font, fontsize=16)
		
		ax.set_xlabel('Predicted class', labelpad=14, fontsize='14', fontproperties=font)
		ax.set_ylabel('True class', labelpad=14, rotation=90, fontsize='14', fontproperties=font)
		
		plt.show()

		return


	def pca_cumulative_graph(self, pca):
		PCs = ['PC'+str(i+1) for i in range(0,22)]
		evr = pca.explained_variance_ratio_
		cumevr = np.cumsum(evr)

		print(evr)
		print(cumevr)

		pcadf = pd.DataFrame({'Principal components': PCs, 'Explained variance ratio': evr})

		font = FontProperties()
		font.set_family(['Times New Roman', 'serif'])
		font.set_size(9)

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4.5))
		ax = sns.barplot("Principal components", "Explained variance ratio", data=pcadf,
                 palette="Blues_d")
		ax.plot(PCs, cumevr, c='#5C6BC0', linestyle='-', marker='o', markersize=2, label='Cumulative explained variance')

		for label in ax.get_xticklabels():
			label.set_fontproperties(font)

		font.set_size(14)

		plt.yticks(np.arange(0.0, 1.0+0.1, 0.1))

		ax.set_ylabel('Explained variance ratio', labelpad=12, fontproperties=font)
		ax.set_xlabel('Principal components', labelpad=12, fontproperties=font)

		plt.grid(alpha=0.3, axis='y', linestyle='--')

		ax.legend()
		plt.show()

		return

	def remove_linear_correlated_features(self, X_train, X_test, columns, display=False):

		df_X_train = pd.DataFrame(X_train, columns=columns)
		df_X_test = pd.DataFrame(X_test, columns=columns)

		del df_X_train['Jitter:DDP']
		del df_X_train['MDVP:Jitter(%)']
		del df_X_train['Shimmer:DDA']
		del df_X_train['MDVP:Shimmer(dB)']
		del df_X_train['Shimmer:APQ5']
		del df_X_train['PPE']

		del df_X_test['Jitter:DDP']
		del df_X_test['MDVP:Jitter(%)']
		del df_X_test['Shimmer:DDA']
		del df_X_test['MDVP:Shimmer(dB)']
		del df_X_test['Shimmer:APQ5']
		del df_X_test['PPE']

		# Nuovi aggiunti
		del df_X_train['MDVP:PPQ']
		del df_X_train['Shimmer:APQ3']
		del df_X_train['MDVP:APQ']
		del df_X_train['MDVP:RAP']
		del df_X_test['MDVP:PPQ']
		del df_X_test['Shimmer:APQ3']
		del df_X_test['MDVP:APQ']
		del df_X_test['MDVP:RAP']

		columns_new = list(columns)
		columns_new.remove('Jitter:DDP')
		columns_new.remove('MDVP:Jitter(%)')
		columns_new.remove('Shimmer:DDA')
		columns_new.remove('MDVP:Shimmer(dB)')
		columns_new.remove('Shimmer:APQ5')
		columns_new.remove('PPE')

		columns_new.remove('MDVP:PPQ')
		columns_new.remove('Shimmer:APQ3')
		columns_new.remove('MDVP:APQ')
		columns_new.remove('MDVP:RAP')

		X_train = df_X_train.values
		X_test = df_X_test.values

		if display:
			print('Len new features:', len(columns_new))
			print(X_train.shape)
			print(X_test.shape)

		return X_train, X_test, columns_new

	def custom_train_test_split(self, X, y, X_individuals, y_individuals, names_individuals, columns, names, test_size=0.25, stratify=True, display=False):

		names_individuals_dict = {ind:name for ind, name in enumerate(names_individuals)}
		# print(names_individuals_dict)

		names_individuals_encoded = np.array([ [i] for i,curr_name in enumerate(names_individuals)], dtype=np.float64)


		X_individuals_temp = np.concatenate((X_individuals, names_individuals_encoded), axis=-1)

		X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_individuals_temp, y_individuals, test_size=test_size, stratify=y_individuals, random_state=self.RANDOM_STATE)

		X_train = list()
		X_test = list()
		y_train = list()
		y_test = list()
		y_test_patients = list()

		for sample, curr_label in zip(X_train_temp, y_train_temp):
			curr_name = int(sample[-1])

			for i,sample in enumerate(X):
				if names[i][:-2] == names_individuals_dict[curr_name]:
					X_train.append(sample)
					y_train.append(curr_label)
		
		for sample, curr_label in zip(X_test_temp, y_test_temp):
			curr_name = int(sample[-1])

			for i,sample in enumerate(X):
				if names[i][:-2] == names_individuals_dict[curr_name]:
					X_test.append(sample)
					y_test.append(curr_label)
					y_test_patients.append(names[i][:-2])

		X_train = np.array(X_train)
		y_train = np.array(y_train)

		X_test = np.array(X_test)
		y_test = np.array(y_test)
		y_test_patients = np.array(y_test_patients)

		if display:
			print('X_train:', X_train.shape)
			unique, counts = np.unique(y_train, return_counts=True)
			print('y_train class distribution\n', np.asarray((unique, counts)).transpose())

			print('X_test:', X_test.shape)
			unique, counts = np.unique(y_test, return_counts=True)
			print('y_test class distribution\n', np.asarray((unique, counts)).transpose())

		X_train_individuals = X_train_temp[:,:-1]
		y_train_individuals = y_train_temp

		X_test_individuals = X_test_temp[:,:-1]
		y_test_individuals = y_test_temp

		if display:
			print('X_train_individuals:', X_train_individuals.shape)
			unique, counts = np.unique(y_train_individuals, return_counts=True)
			print('y_train_individuals class distribution\n', np.asarray((unique, counts)).transpose())

			print('X_test_individuals:', X_test_individuals.shape)
			unique, counts = np.unique(y_test_individuals, return_counts=True)
			print('y_test_individuals class distribution\n', np.asarray((unique, counts)).transpose())

		return X_train, X_test, y_train, y_test, y_test_patients, X_train_individuals, X_test_individuals, y_train_individuals, y_test_individuals

	def dataExploration(self, X, y, X_individuals, y_individuals, columns, names, names_individuals, display=True):

		matplotlib.rcParams["figure.dpi"] = 100
		FONT_SIZE = 14

		font = FontProperties()
		font.set_family(['Times New Roman', 'serif'])
		font.set_size(FONT_SIZE)

		if display:
			print('Number of attributes:', len(X[0]))
			print('Number of samples (original):', len(X))

			unique, counts = np.unique(y, return_counts=True)
			print('Number of samples per class (original)\n', np.asarray((unique, counts)).transpose())

			print('Number of samples (individuals):', len(X_individuals))

			unique, counts = np.unique(y_individuals, return_counts=True)
			print('Number of samples per class (individuals)\n', np.asarray((unique, counts)).transpose())

			print('Dataset shape (original):', X.shape)
			print('Dataset shape (individuals):', X_individuals.shape)

		# scaler1 = StandardScaler(copy=True)
		# scaler1.fit(X)
		# X = scaler1.transform(X)

		df = pd.DataFrame(X, columns=columns)
		df['target'] = y
		df.loc[y==0 ,'target'] = 'Healthy'
		df.loc[y==1 ,'target'] = 'PD'
		df_individuals = pd.DataFrame(X_individuals, columns=columns)

		# --- Scale and PCA both distributions
		# scaler1 = StandardScaler(copy=True)
		# scaler1.fit(X)
		# X = scaler1.transform(X)

		# scaler2 = StandardScaler(copy=True)
		# scaler2.fit(X_individuals)
		# X_individuals = scaler2.transform(X_individuals)

		# pca1 = PCA(n_components=2)
		# pca1.fit(X, y)
		# X = pca1.transform(X)

		# pca2 = PCA(n_components=2)
		# pca2.fit(X_individuals, y_individuals)
		# X_individuals = pca2.transform(X_individuals)
		# ------------

		# ---------- HEAD OF DATASET -----------
		# pd.set_option('display.max_columns', None)
		# print(df.head())
		# --------------------------------------

		# ------ CLASS DISTRIBUTION BAR ------

		# width=0.4

		# unique, counts = np.unique(y, return_counts=True)

		# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4.5))

		# ax.set_ylabel('# of samples', fontproperties=font)

		# # print(np.asarray((unique, counts)).transpose())

		# plt.bar(unique[0], counts[0], width=width, color='#0000ff75', edgecolor='#0000ff90', linewidth=0.5, label='Healthy')
		# plt.bar(unique[1], counts[1], width=width, color='#ff000075', edgecolor='#ff000090', linewidth=0.5, label='Parkinson\'s desease')

		# ax.set_xticks(unique)
		# ax.set_xticklabels(['Healthy', 'Parkinson\'s desease']) #, fontproperties=font

		# for label in ax.get_xticklabels():
		#     label.set_fontproperties(font)

		# # ax.legend()
		# plt.show()

		# width=0.4

		# unique, counts = np.unique(y_individuals, return_counts=True)

		# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4.5))

		# ax.set_ylabel('# of individuals', fontproperties=font)

		# # print(np.asarray((unique, counts)).transpose())

		# plt.bar(unique[0], counts[0], width=width, color='#0000ff75', edgecolor='#0000ff90', linewidth=0.5, label='Healthy')
		# plt.bar(unique[1], counts[1], width=width, color='#ff000075', edgecolor='#ff000090', linewidth=0.5, label='Parkinson\'s desease')

		# ax.set_xticks(unique)
		# ax.set_xticklabels(['Healthy', 'Parkinson\'s desease']) #, fontproperties=font

		# for label in ax.get_xticklabels():
		#     label.set_fontproperties(font)

		# # ax.legend()
		# plt.show()

		# ------------------------------------


		# ---------- BOXPLOT & DENSITY FUNCTIONS (magari boxplot con diversi pazienti per lo stesso boxplot) -------------------

		# Uso le bars di pandas_profiling?
		# pd.set_option('display.max_columns', None)
		# print(df.corr())

		sns.set(color_codes=True)

		# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4.5))

		# ax.set_xlabel('MDVP:Fhi(Hz)',labelpad=8, fontproperties=font)

		# sns.kdeplot(X[y==0,1], shade=True, label='Healthy');
		# sns.kdeplot(X[y==1,1], shade=True, label='PD');

		# ax.legend()

		# # sns.pairplot(df.loc[:, ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
		# # 						'MDVP:Jitter(%)','MDVP:Shimmer',
		# # 						'NHR','HNR','RPDE','DFA','spread2','D2','PPE', 'target']] , hue='target');
		# # sns.pairplot(df.loc[:, ['MDVP:Fo(Hz)', 'spread1', 'PPE', 'target']] , hue='target');

		# sns.pairplot(df.loc[:, ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'target']],
						# hue='target', palette={'Healthy': '#4c72b0', 'PD': '#f57a7a'});

		sns.pairplot(df.loc[:, ['MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP', 'target']],
							hue='target', palette={'Healthy': '#4c72b0', 'PD': '#f57a7a'},
							plot_kws={'alpha': 0.7, 's': 15});

		# sns.pairplot(df.loc[:, ['MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA', 'target']] ,
		# 					hue='target', palette={'Healthy': '#4c72b0', 'PD': '#f57a7a'}); #f57a7a (rosso chiaro) | #dd8452 (orange)

		# sns.pairplot(df.loc[:, ['Shimmer:DDA', 'NHR', 'HNR', 'target']] ,
							# hue='target', palette={'Healthy': '#4c72b0', 'PD': '#f57a7a'});

		# sns.pairplot(df.loc[:, ['spread1', 'PPE', 'target']] ,
		# 					hue='target', palette={'Healthy': '#4c72b0', 'PD': '#f57a7a'});

		plt.show()

		sys.exit()


		# # ------- BOXPLOT

		# scaler = StandardScaler(copy=True)
		# scaler.fit(X)
		# X = scaler.transform(X)

		# df = pd.DataFrame(X, columns=columns)

		# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4.5))
		# sns.boxplot(data=df, orient="v", palette="Set2")

		# ax.set_xticklabels(columns, rotation=60)

		# plt.show()
		# --------


		# ----------------------------------------------------------
		
		# ---------- MOSTRARE SCATTER IN 2 DIMENSIONI -------------------------------------

		# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4.5))

		# ax.set_xlabel('PC1', labelpad=10, fontproperties=font)
		# ax.set_ylabel('PC2', labelpad=10, fontproperties=font)

		# ax.tick_params(length=0)

		# cut_names = [name[:-2] for name in names]
		# cut_names_unique = set(cut_names)
		# colors = plt.cm.rainbow(np.linspace(0,1,len(cut_names_unique)))
		# i=0
		# for name,c in zip(cut_names_unique, colors):
		# 	if y[np.array(cut_names)==name][0] == 0:
		# 		marker = 'o'
		# 	else:
		# 		marker = '^'

		# 	i+=1
		# 	if i%4==0:
		# 		ax.scatter(X[np.array(cut_names)==name,0], X[np.array(cut_names)==name,1], color=c, s=30, alpha=0.7, marker=marker)

		# ax.scatter(X[y==0,0], X[y==0,1], c='#5C6BC0', s=10, alpha=0.7, marker='o', label='Healthy (original)')
		# ax.scatter(X[y==1,0], X[y==1,1], c='#ef5350', s=10, alpha=0.7, marker='o', label='PD (original)')

		# ax.scatter(X_individuals[y_individuals==0,0], X_individuals[y_individuals==0,1], c='#673AB7', s=80, alpha=1, marker='o', label='Healthy (individuals)')
		# ax.scatter(X_individuals[y_individuals==1,0], X_individuals[y_individuals==1,1], c='#D81B60', s=80, alpha=1, marker='o', label='PD (individuals)')

		# ax.legend()
		# plt.show()

		# -----------------------------------------------------

		# ----- SILHOUETTE -----
		
		# labels = [name[:-2] for name in names]

		# # unique, counts = np.unique(labels, return_counts=True)
		# # print(np.asarray((unique, counts)).transpose())

		# scaler = StandardScaler(copy=True)
		# scaler.fit(X)
		# X = scaler.transform(X)

		# # pca = PCA(n_components=22)
		# # pca.fit(X, y)
		# # X = pca.transform(X)

		# X = Normalizer(copy=True).transform(X)

		# silhouette_avg = silhouette_score(X, labels)
		# print(silhouette_avg)

		# random.shuffle(labels)

		# print(labels)

		# silhouette_avg = silhouette_score(X, labels)
		# print(silhouette_avg)

		# ----------------------


		# print(df[:20])
		# print(df.describe())

		# profile = ProfileReport(df)
		# profile.to_file('exploratory_dataset_analysis.html')

		# # profile = df.profile_report(title='Pandas Profiling Report', plot={'histogram': {'bins': 8}})
		# # profile.to_file("output.html")
		# # profile = ProfileReport(df, minimal=True)
		# # profile.to_file("output.html")

		return

	def get_dataset(self):
		X = list()
		y = list()
		columns = list()
		names = list()

		X_individuals = list()
		y_individuals = list()
		names_individuals = list()

		with open('parkinsons.data', 'r', encoding='utf-8') as file:
			reader = csv.reader(file)

			# next(reader)
			header = True
		
			for cols in reader:
				curr_cols = list(cols)
				if header:
					curr_cols.remove('name')
					curr_cols.remove('status')
					columns = list(curr_cols)
					header = False
					continue

				names.append(curr_cols[0])
				del curr_cols[17]
				X.append(curr_cols[1:])
				y.append(int(cols[17]))

		curr_count = 0
		curr_name = names[0][:-2]
		curr_sum = np.zeros(22, dtype=np.float64)

		for i, sample in enumerate(X):
			if names[i][:-2] != curr_name:
				X_individuals.append(list(curr_sum/curr_count))
				y_individuals.append(curr_label)
				names_individuals.append(curr_name)

				# print('this name:',curr_name)
				# print('times:', curr_count)
				# print('average:', list(curr_sum/curr_count))

				# print('---')

				curr_name = names[i][:-2]
				curr_sum = np.zeros(22, dtype=np.float64)
				curr_count = 0

			curr_sum += np.array(sample, dtype=np.float64)
			curr_label = y[i]
			# print(curr_label)
			curr_count += 1

		X_individuals.append(list(curr_sum/curr_count))
		y_individuals.append(curr_label)
		names_individuals.append(curr_name)

		# unique, counts = np.unique(y_individuals, return_counts=True)
		# print(np.asarray((unique, counts)).transpose())

		# print(np.array(X_individuals)[:3])
		# print('# of individuals:', len(np.array(X_individuals)))

		return X, y, X_individuals, y_individuals, names_individuals, columns, names