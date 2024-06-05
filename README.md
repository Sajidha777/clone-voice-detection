## Clone Voice Detection 

#### Overview
This project is aimed at creating lightweight deep learning models to detect deepfake voices, suitable for deployment on mobile devices. It includes fine tuning a pre-trained model called YAMNet, originally trained for classifying environmental sounds. While the initial model architecture is based on MobileNet V1, a significant improvement is achieved by developing a similar but new model using the V2 architecture. The V2 architecture plays a crucial role in enhancing the model's performance. The dataset used in training primarily focuses on phone call scenarios, ensuring the trained models are reliable in detecting spoofed voices during phone conversations.

#### Dataset
The Dataset used is the [ASVSpoof 2021 LA(Logical Access)](https://zenodo.org/records/4837263) dataset. It consists of nearly 200,000 utterances encompassing both authentic and spoofed voices. The spoofed voices are generated using text-to-speech(TTS) and voice conversion algorithms. All the audio samples in the dataset had been transmitted through real telephony networks VoIP or VoIP+PSTN.As a result, they contain artifacts typically associated with telephony transmission, rendering them highly suitable for training models tailored to detecting spoof attacks in phone call scenarios.

#### Notebooks
-  [EDA.ipynb](https://github.com/Sajidha777/clone-voice-detection/blob/main/notebooks/EDA.ipynb) : Exploring the Dataset
-  [yamnet-initial.ipynb](https://github.com/Sajidha777/clone-voice-detection/blob/main/notebooks/yamnet-initial.ipynb) : Fine-tuning the YAMNet model on default(imbalanced) dataset.
-  [data-reduction.ipynb](https://github.com/Sajidha777/clone-voice-detection/blob/main/notebooks/data-reduction.ipynb) : Reducing the data of the majority class systematically as a first step to balancing the dataset. 
-  [augmenting-bonafide.ipynb](https://github.com/Sajidha777/clone-voice-detection/blob/main/notebooks/augmenting_bonafide.py) : Augmenting the data of the minority class to balance the dataset.
-  [yamnet-augmented1.ipynb](https://github.com/Sajidha777/clone-voice-detection/blob/main/notebooks/yamnet-augmented1.ipynb) : Fine-tuning YAMNet model on balanced dataset.
-  [v2model-final.ipynb](https://github.com/Sajidha777/clone-voice-detection/blob/main/notebooks/v2model-final.ipynb) : New architecture i.e replacing v1 with v2. Training with the balanced dataset. The most accurate model.

#### Libraries and their versions
Find it in [requirements.txt](https://github.com/Sajidha777/clone-voice-detection/blob/main/requirements.txt)
