# DALog
**DALog:A Data Augmentation Framework for Log-based Anomaly Detection with Deep Active Learning**

This version is in the form of DALog embedded in Transformer
## Requirements
```
numpy==1.23.4
pandas==1.5.1
python_Levenshtein==0.20.9
torch==1.9.0
tqdm==4.64.1
```
## Log data
HDFS, openStack, and BGL data are all from the [loghub](https://github.com/logpai/loghub). If you are interested in the datasets, please follow the link to submit your access request. We extracted log templates using Drain. The template IDs were used as inputs for anomaly detection.
# Training
To begin training on BGL, run this code:
```
python run.py
```

