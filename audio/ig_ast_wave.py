# Creating wave file of IG while using AST to classify
import torch
import numpy as np
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import librosa

def interpolate_baseline(baseline, inputs, alphas):
    """
    Interpolate between baseline and inputs.
    """
    alphas = alphas[:, None, None]  # Reshape for broadcasting
    return baseline + alphas * (inputs - baseline)

def compute_gradients(model, input_values, target_class_idx):
    """
    Compute gradients of the target class output w.r.t. the model inputs.
    """
    input_values.requires_grad_()  # Enable gradient tracking
    outputs = model(input_values=input_values).logits  # Model forward pass
    target = outputs[:, target_class_idx]
    target.backward()  # Backpropagate to compute gradients
    return input_values.grad.clone().detach()  # Detach the gradients for processing

def integrated_gradients(model, feature_extractor, audio, target_class_idx, baseline=None, num_steps=50, sampling_rate=16000):
    """
    Compute Integrated Gradients for the input audio.
    """
    # Preprocess audio into model input space
    inputs = feature_extractor(raw_speech=audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs["input_values"]  # Extract processed input representation
    
    # Use a zero baseline of the same shape as input_values
    if baseline is None:
        baseline = torch.zeros_like(input_values)
    
    # Generate interpolation steps between baseline and input
    alphas = torch.linspace(0, 1, num_steps)
    interpolated_inputs = interpolate_baseline(baseline, input_values, alphas)

    # Collect gradients at each interpolation step
    gradients = []
    for interp_input in interpolated_inputs:
        interp_input = interp_input.unsqueeze(0)  # Add batch dimension
        grad = compute_gradients(model, interp_input, target_class_idx)
        gradients.append(grad.numpy())
    
    # Average gradients across all steps
    avg_gradients = np.mean(gradients, axis=0)
    
    # Compute Integrated Gradients
    integrated_gradients = (input_values - baseline) * torch.tensor(avg_gradients)
    return integrated_gradients.squeeze(0).numpy()

def classify_and_explain(audio_path, model, feature_extractor, id_to_label, num_steps=50, sampling_rate=16000):
    """
    Classify an audio file and explain the prediction using Integrated Gradients.
    """
    # Load and preprocess the audio
    audio, sr = librosa.load(audio_path, sr=sampling_rate)
    inputs = feature_extractor(raw_speech=audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # Perform classification
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(dim=-1).item()
        predicted_label = id_to_label[predicted_class_idx]
    
    print(f"Predicted class: {predicted_label} (Index: {predicted_class_idx})")

    # Compute Integrated Gradients
    ig_attributions = integrated_gradients(model, feature_extractor, audio, predicted_class_idx, num_steps=num_steps, sampling_rate=sampling_rate)

    # Visualize or return attributions
    return audio, ig_attributions, predicted_label

# Load the fine-tuned AST model and feature extractor
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
model = ASTForAudioClassification.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode
feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)

# Map class IDs to labels (from the model's config)
id_to_label = model.config.id2label

# Path to your MP3 file
audio_path = "gunVsFire.mp3"  # Replace with the path to your audio file

# Classify and compute Integrated Gradients
audio, ig_attributions, predicted_label = classify_and_explain(audio_path, model, feature_extractor, id_to_label)

# Visualize attributions
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(ig_attributions.mean(axis=1), label="IG Attributions (Averaged)")
plt.title(f"Integrated Gradients for {predicted_label}")
plt.legend()
plt.show()

# Output : /media/wav.png
