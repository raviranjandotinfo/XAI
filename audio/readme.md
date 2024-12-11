
## Explainable Artificial Intelligence (XAI) in Audio

Resouce details =>

Hugging face access token of "MIT/ast-finetuned-audioset-10-10-0.4593" model. 

Training Datasets : https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset 

Sample test dataset : /media/gunVsFire.mp3

Notebook = Google colab


Instruction to run the code :-

i) Add Audio spectrogram transformer model token by 
!huggingface-cli login --token TOKEN_NAME

ii) Make sure imported libraries are installed like torch, transformer extra.

iii) Start running file one by one. 

ast_prediction.py = Predicting which class the audio belongs to.

ig_ast_spectogram.py = Applying Integrated gradients to Audio spectogram transformer to generate classified image.

ig_ast_wave.py = Create wave file of IG while using AST to classify.

ig_shap.ipynb = Mask the region on spectrogram image. 


