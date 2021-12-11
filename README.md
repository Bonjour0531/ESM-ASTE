Code and sample data accompanying the paper **Revisiting Aspect-Sentiment-Opinion Triplet Extraction: Detailed Analyses towards a Simple and Effective Span-based Model**

### Instruction 

An example for ESM:

Train 14lap on train dataset, evaluate on dev dataset:

```
python ./esm.py train --config configs/14lap_train.conf
```

Evaluate 14lap on test dataset:
```
python ./esm.py eval --config configs/14lap_eval.conf
```

An example for AT/OT co-extraction baseline BERT+CRF:

```
python -m absa.run_aspect_opinion_extract \
  --vocab_file BERT_DIR/vocab.txt \
  --bert_config_file BERT_DIR/bert_config.json \
  --init_checkpoint BERT_DIR/pytorch_model.bin \
  --do_train \
  --do_predict \
  --data_dir DATA_DIR \
  --train_file 14lap_train.txt \
  --predict_file 14lap_test.txt \
  --train_batch_size 16 \
  --output_dir out
```

### Requirements

- Python 3.5+
- PyTorch 1.1.0+ (tested with version 1.3.1)
- transformers 2.2.0+ (tested with version 2.2.0)
- scikit-learn (tested with version 0.21.3)
- tqdm (tested with version 4.19.5)
- numpy (tested with version 1.17.4)
