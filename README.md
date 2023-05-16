# Transferable Adversarial Distribution Learning: Query-efficient Adversarial Attack against Large Language Models

## 0. Datasets used in our experiments are publicly avaliable.
###  **text classification:**  [yelp](https://huggingface.co/datasets/yelp_polarity) [FakeNews](https://huggingface.co/datasets/BeardedJohn/FakeNews) [AG_News](https://huggingface.co/datasets/ag_news) [DBpedia](https://huggingface.co/datasets/dbpedia_14) [SST-2](https://huggingface.co/datasets/sst2)

### **natural language inference:** [MNLI](https://huggingface.co/datasets/SetFit/mnli)

## 1. Training finetuned models as surrogate models. To help readers quickly reproduce the experimental results, since each fine-tuned model takes about several hours on the 3090GPU, we provide powerful surrogate models trained on different datasets.
Use the following training script to finetune a pre-trained transformer model from HuggingFace:
```
python text_classification.py
```

Well-trained surrogate models are available [here](https://pan.baidu.com/s/106naPV71k8hrdagCidiTOg?pwd=8888).

## 2. Distribution learning on surrogate models
To obtain a strong adversarial distribution based on a finetuned model after running ```text_classification.py``` 
```
python whitebox_attack.py 
```

## 3. black-box attack
After attacking a model, run the following script to query a target model from the optimized adversarial distribution:
```
python evaluate_adv_samples.py 
```
## 4. Acknowledgements
This repository is built using the [FAIR](https://github.com/facebookresearch) repository. We thank C. Guo * et al. for their help. 

