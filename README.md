# AffCon2019 Shared Task
The source code and the details of our system runs for AffCon2019, CL-AFF SHARED TASK: IN PURSUIT OF HAPPINESS
Runs the model on the Happy DB dataset (https://github.com/kj2013/claff-happydb). Please cite the original paper when using the data.


## Autoencoder Model Configuration
	5 hidden layers: 512-256-128-64-128-256-512 
	word dictionary size: 512
	optimizer: SGD
	activation function: LeakyReLU
	loss function: Categorical Cross Entropy
	batch size: 64
	epoch: 30
	normalization: None

## System Run Configuration

	Optimizer: SGD
	Algorithm: SVM
	gamma: 0.1
	loss function: Categorical Cross Entropy
	
	Run No.	label	SVM-kernel 	train data
	1 	social 	linear 		combined  
		agency 	rbf 		original  (dataset 1)
  	2 	social  linear 		combined  
		agency 	rbf 		original  (dataset 2)
  	3 	social 	linear 		combined  
		agency 	rbf  		combined  (dataset 1)
  	4 	social 	linear 		combined 
		agency 	rbf 		combined  (dataset 2)

Note: the agency class was imbalanced in the labeled train set. We created two datasets via random under-sampling 

## Source Code
We split the labeled data into train and test data set by the ratio 8:2. 

1_make_dictionary.py
Creates dictionary file from labeled training data’s moments. Each moment is word-tokenized and stop words were removed using the NLTK 3 package. Then each word-token is counted and the top 1024 (or 512) words are written to the dictionary file.

2_preprocess_labeled.py ( 2_preprocess_unlabeled.py )
Converts the moment into one hot encoding using the dictionary file created above. Each moment is word-tokenized and stop words were removed using the NLTK package in the same manner as above. Then the agency label and social label is appended at the end (0 for no, 1 for yes). For example, for dictionary size of 512 the structure of processed data looks like below.

512 (OneHot encoding of moment) + 1 (agency label) + 1 (social label)

3_performance_check.py
Checks the precision of selected instances (top 3% smallest loss) after training the One Class Autoencoder. Everything is called inside the 'my_run' function.

'label_type': 0 for using agency as the label, 1 for social
'train_val': 0 trains the autoencoder only with instances with label 'no', 1 trains the autoencoder only with instances with label 'yes'.
'load_train' function loads the labeled training data of the given configuration ('label_type', 'train_val'). 'load_test' function loads the labeled test data.
 
'autoencoder_model' function returns the Keras autoencoder model. Then the autoencoder model is trained with both data and label as the train data. 

After the model is trained, test data is inserted to the trained model and 'pct'% of instances with smallest loss is chosen. 3 loss functions are provided in this file.

'myDistance' function : l1 distance
'myMse' function : mse loss
'mycross_entropy' function : cross entropy loss

The chosen instances are presumed to have the label of 'train_val'.

4_labeling.py
Trained autoencoder model of given configuration ('label_type', 'train_val') from '3_performance_check.py' is called. Then the unlabled training data is insterted to the model and 'pct'% instances of smallest loss values are presumed to have the label 'train_val' and is written to file.

5_combine.py
Combines the labeled training data with picked unlabeled data from '4_labeling.py'.

6_knn_performance.py (6_rf_performance.py, 6_svm_performance.py, 6_xgboost.py)
Compares the performance of algorithms between training with original data (labeled training data) and training with combined data (labeled training data + unlabeled picked data). Algorithms used for testing are KNN, SVM, and Random Forest from sklearn package and XGboost from XGBoost package.

7_prepare.py
Processes unlabeled test data using the dictionary in the same fashion as '2_preprocess_unlabeled.py'.

7_answer.py
Labels the instances of unlabeled test data using the algorithms mentioned in '6_....py' files.

