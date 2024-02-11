STT_HOST = 'https://coqui.gateway.scarf.sh'
STT_HOST_AHOLAB = 'https://aholab.ehu.eus/~xzuazo/models'

STT_MODELS = {

    'eu': {
        'name': 'Basque STT v0.1.8',
        'language': 'Basque',
        'version': 'v0.1.8',
        'creator': 'ITML',
        'acoustic': f'{STT_HOST_AHOLAB}/Basque STT v0.1.7/model.tflite',
        'scorer': f'{STT_HOST_AHOLAB}/Basque STT v0.1.7/kenlm.scorer',
        # hyperparameters
        'lm_alpha': 1.44,
        'lm_beta': 4.99,
    }

}