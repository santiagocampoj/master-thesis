# STT Project

## Introduction

This project focuses on Speech-to-Text (STT) processing using the Deep Speech model. It aims to transcribe audio files and generate metrics to evaluate the model's performance across different databases. The project primarily deals with Spanish and Basque, which are low-resource languages. Two versions of the Basque model, v1.7 and v1.8, are used for processing.

## Project Structure

The project is organized into different language directories with further subdivisions for various model versions:

- `Language/`
  - `Spanish/`
  - `Basque/`
    - `v1.7/`
    - `v1.8/`

### Key Components

- `logger_config.py`: Sets up custom logging for audio processing.
- `model_config.py`: Contains configuration settings for the STT models, including model URLs and versions.
- `stt_class.py`: The primary class for handling STT operations. It includes methods for model downloading, audio processing, and metrics computation.
- `utils.py`: Contains utility functions for data preprocessing and audio processing.
- `main.py`: The main script to run the STT process, integrating components from other modules.

## Utilization

The project involves several key steps:

1. **Audio Processing**: Audio files are transcribed using the Deep Speech model.
2. **Metrics Computation**: The project calculates Word Error Rate (WER) and other relevant metrics to evaluate the model's performance.
3. **Support for Multiple Languages**: Customization to work with both Spanish and Basque, offering insights into STT performance in low-resource languages.

## How to Run

1. Install required dependencies (list dependencies if necessary).
2. Set the correct paths in `model_config.py`.
3. Run `main.py` with the appropriate arguments for the audio and text files.

## Contributing

Contributions to the project are welcome once it is done. Please ensure to follow the code structure and naming conventions.

# License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

