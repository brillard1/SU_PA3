# Audio Deepfake Detection

This repository contains the code and compilation instructions for Speech Programming Assignment 3.

We run the model using `—custom` argument for custom dataset, by default it takes FOR dataset. And we change the model_path to '../checkpoints/Best_LA_model_for_DF.pth' for DF setting.

The evaluation scores on custom dataset can be compiled using the below command,
```
CUDA_VISIBLE_DEVICES=0 python custom_main.py --track=LA --is_eval --eval --custom --model_path='../checkpoints/model_LA.pth' --eval_output='custom_scores.txt'
```

The below command fine-tunes the model on FOR dataset using weighted cross-entropy (WCE) loss,
```
CUDA_VISIBLE_DEVICES=0 python custom_main.py --model_path="../checkpoints/LA_model.pth" --track=LA --lr=0.000001 --batch_size=4 --num_epochs=5 --loss=WCE
```

The updated weights are used to evaluate FOR dataset.
```
CUDA_VISIBLE_DEVICES=0 python custom_main.py --track=LA --is_eval --eval --model_path='../checkpoints/epoch_5.pth' --eval_output='for_finetuned_scores.txt'
```
- Remove the `--custom` argument to change the dataset mode.

Compile the results for custom dataset from fine-tuned model using the below command.

```
CUDA_VISIBLE_DEVICES=0 python custom_main.py --track=LA --is_eval --eval --custom --model_path='../checkpoints/epoch_5.pth' --eval_output='custom_finetuned_scores.txt'
```