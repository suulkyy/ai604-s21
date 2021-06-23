# AI604-Spring2021 Assignment #3 - Self-Supervised Learning

## Implementation: Self-Supervised Models

### Completing the Forward-propagation of the Model

Please refer [here](https://github.com/suulkyy/ai604-s21/blob/main/hw3-20214196/simsiam/model_factory.py)

### Contrastive Loss

Please refer to the function 'contrastive_loss' in [here](https://github.com/suulkyy/ai604-s21/blob/main/hw3-20214196/simsiam/criterion.py)

### Align Loss

Please refer to the function 'align_loss' in [here](https://github.com/suulkyy/ai604-s21/blob/main/hw3-20214196/simsiam/criterion.py)

### Updating the Momentum Encoder

Please refer to the 'Question 1.4' in [here](https://github.com/suulkyy/ai604-s21/blob/main/hw3-20214196/main.py)

## Training the Model

### K-Nearest Neighbor Classification

Please refer to 'KNN' in [here](https://github.com/suulkyy/ai604-s21/blob/main/hw3-20214196/simsiam/validation.py)

### Training

1. Self-Supervised Training

[(SimCLR)](https://arxiv.org/pdf/2002.05709.pdf) 'python main.py --loss_type contrastive --use_momentm_encoder False --use_predictor False --stop_gradient False
[(MoCo)](https://arxiv.org/pdf/1911.05722.pdf) 'python main.py --loss_type contrastive --use_momentum_encoder True --use_predictor False --stop_gradient True
[(BYOL)](https://arxiv.org/pdf/2006.07733.pdf) 'python main.py --loss_type align --use_momentum_encoder True --use_predictor True --stop_gradient True
[(SimSiam)](https://arxiv.org/pdf/2011.10566.pdf) 'python main.py --loss_type align --use_momentum_encoder False --use_predictor True --stop_gradient True

2. Linear Evaluation

'''(bash)
python main_lincls.py --pretrained [your checkpoint path]
'''

