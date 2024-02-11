STT_HOST = 'https://coqui.gateway.scarf.sh'
STT_HOST_AHOLAB = 'https://aholab.ehu.eus/~xzuazo/models'

STT_MODELS = {
    'whisper': {
        'name': 'Whisper Basque v0.0.1',
        'language': 'Basque',
        'language_code': 'eu',
        'version': 'v0.0.1',
        'creator': 'Xabier Zuazo',
        'acoustic': '/home/aholab/santi/Documents/audio_process/Language/models/Whisper STT v0.0.1/zuazo-whisper-medium-eu.pt',
        'language_model': '/home/aholab/santi/Documents/audio_process/Language/models/Whisper STT v0.0.1/5gram.bin',
    },
    'eu': {
        'name': 'Basque STT v0.1.7',
        'language': 'Basque',
        'version': 'v0.1.7',
        'creator': 'ITML',
        'acoustic': f'{STT_HOST_AHOLAB}/Basque STT v0.1.7/model.tflite',
        'scorer': f'{STT_HOST_AHOLAB}/Basque STT v0.1.7/kenlm.scorer',
    }
}
