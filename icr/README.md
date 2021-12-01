# ICR


## Data splitting

```
python ./split_dataset.py --inputPath ~/datasets/icr-finetune --gtFile ~/datasets/icr-finetune/labels.txt --outputPath ~/datasets/icr-finetune-split  --train_percentage .8
```

## Finetuning

Prepare finetuned dataset for both traing and testing

This will read in the `labels.txt' file and generate an `gtFile`, this generator supports nested folders for dataset creation


```
python3 create_lmdb_dataset-gen.py --inputPath ~/datasets/icr-finetune/train --gtFile ~/datasets/icr-finetune/train/labels-agro.txt --outputPath data_lmdb_release/training/finetune

python3 create_lmdb_dataset-gen.py --inputPath ~/datasets/icr-finetune/test --gtFile ~/datasets/icr-finetune/test/labels-agro.txt --outputPath data_lmdb_release/validation/finetune

```

### Start finetuning

```
 CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data finetune --batch_ratio 1. --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --batch_max_length 48 --imgH 32 --imgW 100 --batch_size 192 --sensitive --PAD --data_filtering_off --FT --saved_model ../models/icr/TPS-ResNet-BiLSTM-Attn-case-sensitive-ft/best_accuracy.pth

```

### Evaluation

```
CUDA_VISIBLE_DEVICES=-1 python test-icr.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder  ~/datasets/debug-boxes --saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth --sensitive --imgH 32 --imgW 100
```


