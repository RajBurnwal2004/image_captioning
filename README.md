# Multimodal Image Captioning System

An image-captioning system that combines pretrained **Xception CNN features** with an **LSTM language decoder** to generate natural-language captions for images. The model was developed and evaluated on the **Flickr8k** dataset using a leak-free train-validation-test split.

## Overview

The system follows an encoder-decoder architecture:

```text
Image
  ↓
Pretrained Xception CNN
  ↓
2048-D image feature
  ↓
Dense Projection
  ↓
256-D image representation
  ↓
        ┌──────────────┐
        │ Element-wise │
        │    Fusion    │
        └──────┬───────┘
               ↑
Caption → Tokenization → Embedding → LSTM
                                      ↓
                                  256-D text
                                  representation
                                      ↓
                                  Dense Layer
                                      ↓
                              Softmax Vocabulary
                                      ↓
                              Next-word prediction
