How to run run the baseline model

The baseline model (MR-CNN by Niu et.al) is based on https://github.com/Raschka-research-group/coral-cnn/blob/master/model-code/afad-ordinal.py

To run the file:

Give as arguments the path to the directory for images, log_directory,  training csv file, validation csv file, test_csv file.

Eg:

run_model.py  --image_dir --path to where logs are to be stored --training csv_data path
--validation csv file path, --training csv file path

In case of doubts; please run: run_model.py --help

