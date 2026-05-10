# Self-Attention From Scratch

This is a personal project dedicated to understanding and practicing the core concepts of the Attention mechanism. The repository contains from-scratch implementations of various attention architectures to build a solid foundational understanding of how modern Transformer models work under the hood.

## Features

This project includes implementations of the following mechanisms using **PyTorch**:

- **Self-Attention**: The standard scaled dot-product attention mechanism.
- **Masked Self-Attention**: Self-attention with causal masking (look-ahead mask), typically used in decoder blocks to prevent future tokens from being attended to.
- **Multi-Head Attention**: Extending single-head attention to multiple heads to capture different representational subspaces simultaneously.
- **Encoder-Decoder Attention**: Cross-attention mechanism where queries come from the decoder and keys/values come from the encoder outputs.

## Requirements

- Python 3.12
- PyTorch

## Usage

This repository is mainly for educational purposes and hands-on practice. You can explore the `.py` files to see the step-by-step mathematical implementations of each attention block using PyTorch tensor operations.
