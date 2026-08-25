# Multimodal Image Captioning System

An image-captioning system combining pretrained Xception CNN features with an LSTM language decoder to generate natural-language captions for images from the Flickr8k dataset.

## Overview

The project follows an encoder-decoder architecture:

Image → Xception CNN → 2048-D feature → Dense(256)

Caption → Tokenization → 256-D Embedding → LSTM(256)

The 256-D image and text representations are fused using element-wise addition and passed through dense and softmax layers to predict the next word in the caption.

## Dataset

Flickr8k was divided into three non-overlapping splits:

- Training: 6,000 images
- Validation: 1,000 images
- Test: 1,000 images

Each image contains 5 human-written reference captions.

## Image Encoder

A pretrained Xception model was used as the image feature extractor.

- Input image size: 299 × 299
- Classification head removed
- Global average pooling applied
- Output representation: 2,048 dimensions

The Xception features were precomputed and stored so that the CNN did not need to perform a forward pass repeatedly during captioning-model training.

The 2,048-D image representation was projected to 256 dimensions using a fully connected layer.

## Text Decoder

Captions were tokenized using a Keras Tokenizer.

- Vocabulary size: 7,577 words
- Word embedding dimension: 256
- LSTM units: 256
- Maximum sequence length: 33 tokens
- Dropout: 0.5

The LSTM processes the caption sequentially and produces a 256-D hidden representation of the caption prefix.

## Multimodal Fusion

The image and text representations were both projected to 256 dimensions and combined using element-wise addition.

256-D image representation
+
256-D text representation
↓
256-D fused representation
↓
Dense(256)
↓
Softmax over vocabulary

The final softmax layer predicts the probability of each vocabulary word being the next word in the caption.

## Training

Captions were converted into autoregressive next-word prediction sequences.

For example:

start → a

start a → dog

start a dog → is

start a dog is → running

This produced approximately 276K sequence-level training samples from the training captions.

The model was trained using:

- Optimizer: Adam
- Loss: Sparse Categorical Cross-Entropy
- Batch size: 32
- Maximum sequence length: 33
- Training epochs: 10
- Validation loss for model selection

During training, the ground-truth caption prefix was provided to the model while predicting the next word.

## Caption Generation

During inference, captions were generated autoregressively using greedy decoding.

The process starts with the special `start` token:

start
↓
predict next word
↓
append predicted word
↓
predict next word
↓
continue
↓
end

At every step, the word with the highest predicted probability was selected.

Generation stopped when the `end` token was produced or the 33-token maximum length was reached.

Precomputed Xception features were reused during testing to avoid unnecessary CNN computation.

## Evaluation

The final model was evaluated on 1,000 held-out test images.

Each generated caption was compared against the 5 human-written reference captions associated with the corresponding test image.

Evaluation metrics:

- BLEU-1: 0.425
- BLEU-2: 0.1955
- BLEU-3: 0.1024
- BLEU-4: 0.0520
- METEOR: 0.245

BLEU measures n-gram overlap between generated and reference captions, while METEOR provides an additional word-level caption similarity measure.

## Key Features

- Pretrained Xception-based visual encoder
- 2,048-D precomputed image representations
- 256-D multimodal representation
- LSTM-based autoregressive language decoder
- 7,577-word vocabulary
- 276K+ sequence-level training samples
- Separate train, validation, and test splits
- Validation-based model checkpoint selection
- Greedy autoregressive caption generation
- BLEU and METEOR evaluation

## Technologies

Python  
TensorFlow / Keras  
Xception  
LSTM  
NumPy  
NLTK  
PIL  
Matplotlib  
Google Colab  
Kaggle

## Future Improvements

- Replace the LSTM decoder with a Transformer-based decoder
- Introduce visual attention over spatial CNN features
- Compare element-wise addition with concatenation for multimodal fusion
- Use beam search instead of greedy decoding
- Fine-tune the visual encoder
- Evaluate on larger image-captioning datasets
- Add qualitative error analysis and failure-case analysis
