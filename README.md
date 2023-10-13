TsKI
=====

This repo contains code for Tabular Reasoning via Two-stage Knowledge Injection.

Setup
-------
Run the following command to install the dependency packages.
```bash
pip install requirements.txt
```

Data Generation
-------
Run the following command to install the dependency packages.
```bash
cd generate_lf
python generate_lf.py --output OUTPUT_DIR --table_id_file ./table_id.txt --table_source_folder TABLE_FOLDER
```
The ```TABLE_FOLDER``` can be found in the repo [https://github.com/wenhuchen/Table-Fact-Checking](https://github.com/wenhuchen/Table-Fact-Checking).

Training and Evaluating
-------
Run the following command to perform pre-training and fine-tuning respectively.
```bash
bash run_pretrain.sh
```
```bash
bash run_finetune.sh
```
For the model evaluation, you can remove the ```--do-train``` in all scripts and change the ```--model_name_or_path``` to the trained model paths.
