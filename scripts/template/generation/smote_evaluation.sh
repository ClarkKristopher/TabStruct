python -m src.experiment.run_experiment \
	--pipeline 'generation' \
	--model 'smote' \
	--eval_only \
	--dataset "mfeat-fourier" \
	--test_size 0.2 \
	--valid_size 0.1 \
	--generator_tags "SMOTE-generation" \
	--tags 'SMOTE-evaluation'