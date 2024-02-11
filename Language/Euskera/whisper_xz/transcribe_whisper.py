import argparse
import whisper
import model_config_xz
from whisper_evaluate import set_lm_options, parse_transcribe_options

def transcribe(audio_path, lm_alpha, lm_beta):
    """
    Transcribe the given audio file using the custom Whisper model.

    Parameters:
        audio_path (str): Path to the audio file.
        lm_alpha (float): KenLM Language Model weight.
        lm_beta (float): KenLM word insertion weight.

    Returns:
        str: The transcribed text.

    Usage:
        python transcribe_whisper.py <audio file>

    Example of usage:
        python transcribe_whisper.py /home/aholab/santi/Documents/audio_process/Test/Language/Euskera/whisper/xzuazo/whisper-lm-gitlab/U0001077.wav
    """
    
    model_config_data = model_config_xz.STT_MODELS['whisper']
    model_path = model_config_data['acoustic']
    model = whisper.load_model(model_path)

    # lm options
    args = argparse.Namespace(lm_alpha=lm_alpha, lm_beta=lm_beta)
    set_lm_options(args)

    # transcribe
    transcribe_options = parse_transcribe_options(args)
    print(f"Transcribe options: {transcribe_options}")
    
    result = model.transcribe(audio_path, **transcribe_options)

    return result['text']

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using a custom Whisper model.")
    parser.add_argument("audio_path", type=str, help="Path to the audio file to transcribe.")
    parser.add_argument("--lm_alpha", type=float, default=0.33582368603855817, help="KenLM Language Model weight.")
    parser.add_argument("--lm_beta", type=float, default=0.6882556478819416, help="KenLM word insertion weight.")
    args = parser.parse_args()

    transcription = transcribe(args.audio_path, args.lm_alpha, args.lm_beta)
    print("\nTranscription:\n", transcription)

if __name__ == "__main__":
    main()
