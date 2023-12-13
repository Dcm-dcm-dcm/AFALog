# AFALog
**AFALog: A General Augmentation Framework for Log-based Anomaly Detection with Active Learning**

This version is in the form of AFALog embedded in Transformer
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
# Citation
Please cite if you use the data or code in this repo.
```
@INPROCEEDINGS{10301228,
  author={Duan, Chiming and Jia, Tong and Cai, Huaqian and Li, Ying and Huang, Gang},
  booktitle={2023 IEEE 34th International Symposium on Software Reliability Engineering (ISSRE)}, 
  title={AFALog: A General Augmentation Framework for Log-based Anomaly Detection with Active Learning}, 
  year={2023},
  volume={},
  number={},
  pages={46-56},
  doi={10.1109/ISSRE59848.2023.00068}}
```
