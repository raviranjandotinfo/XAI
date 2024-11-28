# Predicting which class the audio belongs to using Audio spectogram transformer

import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import librosa

def classify_audio(file_path, model, feature_extractor, id_to_label, target_sr=16000):
    """
    Classify audio by converting it to the required input format and passing it through the AST model.
    """
    # Load and preprocess the raw audio file
    audio, sr = librosa.load(file_path, sr=target_sr)  # Resample to 16 kHz
    inputs = feature_extractor(raw_speech=audio, sampling_rate=target_sr, return_tensors="pt", padding=True)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(dim=-1).item()
        predicted_label = id_to_label[predicted_class_idx]

    return predicted_class_idx, predicted_label

# Load the fine-tuned AST model and feature extractor
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
model = ASTForAudioClassification.from_pretrained(model_name)
feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)

# Map class IDs to labels (from the model's config)
id_to_label = model.config.id2label

# Path to your MP3 file
audio_path = "/media/gunVsFire.mp3"  #  audio file name

# Classify the audio file
predicted_class_idx, predicted_label = classify_audio(audio_path, model, feature_extractor, id_to_label)

# Print the classification result
print(f"Predicted class index: {predicted_class_idx}")
print(f"Predicted label: {predicted_label}")

# Output : Predicted class index: 427
#Predicted label: Gunshot, gunfire
