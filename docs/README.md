## Documentation

- The dockerfile is configured to install Quarto and can be run from the quarto CLI
- To generate html reports from model outputs, download the json results from storage and run the following command:

```bash
quarto render read_pyspacer_results.qmd - P valresult_filepath:valresult_ba_2024-04-16_16-09-03.json      
```
- The output will be saved in the same directory as the json file with the same name and a .html extension


## Pre-processing

- This pre-processing is a one-off script after the feature vectors dump from CoralNet. It is not necessary to run this script again unless the feature vectors are updated.

- Note this will traverse an entire S3 directory to create a parquet file. There may be other approaches that are more suitable for a development environment.

- The script is `dask_preprocessing.qmd`

