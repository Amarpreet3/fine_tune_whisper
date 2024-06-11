# fine_tune_whisper

# Fine-Tune Whisper for Multilingual ASR

This repository contains a script for fine-tuning the Whisper model for any multilingual Automatic Speech Recognition (ASR) dataset using Hugging Face 🤗 Transformers.

## Introduction

Whisper is a pre-trained model for automatic speech recognition (ASR) published in [September 2022](https://openai.com/blog/whisper/) by the authors Alec Radford et al. from OpenAI. Unlike many of its predecessors, such as [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477), which are pre-trained on unlabelled audio data, Whisper is pre-trained on a vast quantity of labelled audio-transcription data, 680,000 hours to be precise. This is an order of magnitude more data than the unlabelled audio data used to train Wav2Vec 2.0 (60,000 hours).

## Features

- Fine-tune Whisper for multilingual ASR tasks.
- Utilize the Hugging Face 🤗 Transformers library for model training and evaluation.
- Support for various datasets and customization options.

## Requirements

- Python 3.6 or higher
- `transformers` library from Hugging Face
- `datasets` library from Hugging Face
- `torch` for PyTorch

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

## Usage

1. **Prepare your dataset:** Ensure your dataset is in a compatible format for training. You may need to preprocess the data to match the expected input format of the Whisper model.

2. **Run the fine-tuning script:**

    ```bash
    python fine_tune_whisper.py --dataset_path /path/to/your/dataset --output_dir /path/to/save/model
    ```

    Replace `/path/to/your/dataset` with the path to your dataset and `/path/to/save/model` with the directory where you want to save the fine-tuned model.

3. **Customize parameters:** The script allows various parameters to be customized, such as the learning rate, batch size, number of epochs, etc. You can modify these directly in the script or pass them as arguments.

## Script Overview

The `fine_tune_whisper.py` script includes the following sections:

- **Introduction:** Overview of the Whisper model and its capabilities.
- **Setup:** Import necessary libraries and setup configurations.
- **Data Preparation:** Load and preprocess the dataset for training.
- **Model Fine-Tuning:** Configure and fine-tune the Whisper model.
- **Evaluation:** Evaluate the performance of the fine-tuned model on a validation set.
- **Saving the Model:** Save the fine-tuned model for future use.

## References

- [Whisper Blog Post](https://openai.com/blog/whisper/)
- [Wav2Vec 2.0 Paper](https://arxiv.org/abs/2006.11477)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or enhancements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
