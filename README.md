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
	1 	social 	linear 		combined data 
		agency 	rbf 		original data
  	2 	social  linear 		combined data 
		agency 	rbf 		original data
  	3 	social 	linear 		combined data 
		agency 	rbf  		combined data
  	4 	social 	linear 		combined data
		agency 	rbf 		combined data

	social	train data		kernel	
	1	combined		linear	
	2	combined		linear	
	3	combined		linear	
	4	combined		linear	
						
	agency	train data		kernel	
	1	original (dataset 1)	rbf	
	2	original (dataset 2)	rbf	
	3	combined (dataset 1)	rbf	
	4	combined (dataset 2)	rbf	

Note: the original agency 
