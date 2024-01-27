# Text-Guided Saliency  Prediction Model

## Requirements
* transformers $\rightarrow$ BLIP
* scipy
* torch
* ...

## File Structure

- `config.py`: model parameters
- `utilities.py`: Dataset loader for both SALICON and SJTU-TIS
- `TransformerEncoder.py`: code from TranSalNet, no need for change
- `resnet.py`: resnet loader, no need for change
- `requirements.txt`: contains the dependencies of this project.
- `model.py`: all menetioned model implementation and metrics implementation
- `main.py`: the main function called to train and evaluate the models

## Training
3 methods are proposed in the paper and the file provided implements the last one, that is the one inserts txt features using contrastive loss. If the previous 2 methods are to be utilized, simple modification could be done by setting the contrastive loss coefficient to 0 and add cross attention layers at the places mentioned in the paper. (e.g. for place 1, to be inserted at line 594 - 599, for place 2, to be inserted at line 680)

```text
usage: main.py

[-gpu_id] [--dataset] [--img_type] [--finetune] [--seed] [--rank_pair]             
```
the model should firstly be pre-trained on SALICON and then use the finetune parameter to load the pretrained model and further finetune on the SJTU-TIS dataset.

Encoder and Decoder needs to be unfreezed before training for pretraining on SALICON.

The result will be recorded under ./log and models will be saved under ./model