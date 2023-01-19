# BASELINE MODELS
for MODEL_NAME in  SparsePCA16 SparsePCA32 PPCA16 PPCA32
do
    kedro run --pipeline baseline_experiment --params model_name:$MODEL_NAME
done

# VAE MODELS
for MODEL_NAME in  LinearVAE LassoVAE SparseVAE
do
    kedro run --pipeline experiment --params model_name:$MODEL_NAME
done
