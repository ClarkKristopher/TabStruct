python -m src.experiment.run_experiment \
    --pipeline 'generation' \
    --model 'smote' \
    --generation_only \
    --dataset 'mfeat-fourier' \
    --test_size 0.2 \
    --valid_size 0.1 \
    --tags 'SMOTE-generation'