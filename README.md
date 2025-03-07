Transformer Implementation: Attention Is All You Need

Overview

This repository contains a complete implementation of the Transformer model, as introduced in the seminal paper "Attention Is All You Need". The model is built from scratch using PyTorch, demonstrating the architecture's core components, including self-attention, positional encoding, and multi-head attention.

Features

Full implementation of the Transformer model

Modular and well-documented codebase

Customizable hyperparameters for experimentation

Supports training from scratch on NLP tasks

Installation

To run this implementation, you need Python 3.8+ and PyTorch. Install the required dependencies using:

pip install -r requirements.txt

Usage

Training the Model

Run the following command to train the Transformer model:

python train.py --config config.yaml

Evaluating the Model

To evaluate on a test dataset:

python evaluate.py --checkpoint checkpoint.pth

Architecture

The Transformer model consists of the following key components:

Multi-Head Self-Attention: Allows the model to focus on different parts of the input sequence simultaneously.

Positional Encoding: Injects information about token order into the model.

Feed-Forward Networks: Adds non-linearity and learning capacity.

Layer Normalization & Residual Connections: Stabilizes training and improves convergence.

Step-by-Step Breakdown of Transformer Model (14 Steps)

Input Embeddings - The input text is tokenized and converted into vector embeddings.

Positional Encoding - Adds information about the order of words in the sequence.

Input to Encoder - The embedding and positional encoding are summed and passed into the encoder.

Multi-Head Self-Attention (Encoder) - Computes attention scores over the entire sequence.

Add & Norm (Encoder) - Adds residual connections and applies layer normalization.

Feed-Forward Network (Encoder) - Applies a feed-forward network to each position.

Add & Norm (Encoder) - Another layer normalization step.

Stacked Encoders - Multiple encoder layers process the input sequentially.

Input to Decoder - The target sequence is embedded and positionally encoded.

Masked Multi-Head Self-Attention (Decoder) - Prevents looking at future words during training.

Multi-Head Attention (Decoder-Encoder Attention) - Attends to the encoder's output.

Feed-Forward Network (Decoder) - Applies transformations similar to the encoder.

Final Linear Layer & Softmax - Converts the output to probability distributions over the vocabulary.

Output Prediction - The highest probability token is selected as the final output.

 



References

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). "Attention Is All You Need." Advances in Neural Information Processing Systems.

 

Contribution

Contributions are welcome! Please open an issue or submit a pull request with any improvements.

 

