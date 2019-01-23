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

Note: the agency class was imbalanced in the labeled train set. We created two datasets via random under-sampling. 
