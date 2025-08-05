# Adding stage in dvc.yaml
dvc stage add -n data_ingestion -d src/data_ingestion.py -o data/raw python3 src/data_ingestion.py