

# Training and Inference notes

https://github.com/NielsRogge/Transformers-Tutorials

https://www.kaggle.com/ammarnassanalhajali/fine-tuning-layoutlmv2-on-funsd/notebook

https://nanonets.com/blog/layoutlm-explained/


Same thing copy/paste with small changes

https://towardsdatascience.com/fine-tuning-transformer-model-for-invoice-recognition-1e55869336d4
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/True_inference_with_LayoutLMv2ForTokenClassification_%2B_Gradio_demo.ipynb

https://www.kaggle.com/ammarnassanalhajali/fine-tuning-layoutlmv2-on-funsd


Run funsd Training

```
python ./run_funsd.py --do_eval=True --do_predict=True --do_train=True --evaluation_strategy=epoch --fp16=True --load_best_model_at_end=True --max_train_samples=1000 --model_name_or_path=microsoft/layoutlmv2-base-uncased --num_train_epochs=30   --report_to=wandb --save_strategy=epoch --save_total_limit=1 --warmup_ratio=0.1  --output_dir ./results
```

python examples/run_funsd.py \
        --model_name_or_path microsoft/layoutlm-base-uncased \
        --output_dir /tmp/test-ner \
        --do_train \
        --do_predict \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --load_best_model_at_end \
        --save_total_limit=1


python examples/run_funsd.py \
        --model_name_or_path microsoft/layoutlm-base-uncased \
        --output_dir /tmp/test-ner \
        --do_predict \
        --max_steps 1000 \
        --warmup_ratio 0.1 \
        --fp16 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --load_best_model_at_end 
