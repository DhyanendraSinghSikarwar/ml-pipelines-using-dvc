# Adding stage in dvc.yaml
dvc stage add -n data_ingestion -d src/data_ingestion.py -o data/raw python3 src/data_ingestion.py
# command to add new stage with previous stage dependency 
dvc stage add -n data_preprocessing -d src/data_preprocessing.py -d data/raw -o data/processed python3 src/data_preprocessing.py
#  command to run the pipeline
dvc repro
# to visualize all the stages
dvc dag 
# to add metrics as well
dvc stage add -n model_evaluation -d src/model_evaluation.py  -d model.pkl --metrics metrics.json python3 src/model_evaluation.py 
# to check the metrics
dvc metrics show