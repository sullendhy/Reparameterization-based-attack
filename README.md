# Reparameterizetion-based attack with multi-objective geometric loss

## 0. Install Dependencies and Data Download:
```
conda install -c huggingface transformers
pip install datasets
```

## 1. Training finetuned models
Use the following training script to finetune a pre-trained transformer model from HuggingFace:
```
python text_classification.py --data_folder <data_dir> --dataset <dataset_name> --model <model_name> --finetune True
```

## 2. Attacking a finetuned model
To attack a finetuned model after running ```text_classification.py``` or from the TextAttack library:
```
python whitebox_attack.py 
```


## 3. Evaluating transfer attack and black-box attack
After attacking a model, run the following script to query a target model from the optimized adversarial distribution:
```
python evaluate_adv_samples.py 
```
