# AffCon2019 Shared Task
The source code and the details of our system runs for AffCon2019, CL-AFF SHARED TASK: IN PURSUIT OF HAPPINESS

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

	Run 1. social trained with svm-linear and combined data & agency trained with svm rbf and original data
  	Run 2. social svm linear combined & agency svm rbf original
  	Run 3. social svm linear combined & agency svm rbf combined
  	Run 4. social svm linear combined & agency svm rbf combined

	social	optimizer	loss	svm_train	kernel	gamma
	1	sgd	categorical_cross_entropy	combined	linear	0.1
	2	sgd	categorical_cross_entropy	combined	linear	0.1
	3	sgd	categorical_cross_entropy	combined	linear	0.1
	4	sgd	categorical_cross_entropy	combined	linear	0.1
						
	agency	optimizer	loss	svm_train	kernel	gamma
	1	sgd	categorical_cross_entropy	original (split 1)	rbf	0.1
	2	sgd	categorical_cross_entropy	original (split 2)	rbf	0.1
	3	sgd	categorical_cross_entropy	combined (split 1)	rbf	0.1
	4	sgd	categorical_cross_entropy	combined (split 2)	rbf	0.1
