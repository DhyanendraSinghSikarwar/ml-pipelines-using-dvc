# Adding stage in dvc.yaml
dvc stage add -n data_ingestion -d src/data_ingestion.py -o data/raw python3 src/data_ingestion.py
# command to add new stage with previous stage dependency 
dvc stage add -n data_preprocessing -d src/data_preprocessing.py -d data/raw -o data/processed python3 src/data_preprocessing.py
#  command to run the pipeline
dvc repro
# to visualize all the stages
dvc dag 