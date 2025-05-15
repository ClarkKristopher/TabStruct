python -m src.experiment.run_experiment \
	--model 'knn' \
	--save_model \
	--dataset 'adult' \
	--test_size 0.2 \
	--valid_size 0.1 \
	--numerical_transform 'quantile' \
	--reg_test \
	--tags 'dev'
