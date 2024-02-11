STT_HOST = 'https://coqui.gateway.scarf.sh'
STT_HOST_AHOLAB = 'https://aholab.ehu.eus/~xzuazo/models'

STT_MODELS = {

    'es': {
        'name': 'Spanish STT v0.0.1',
        'language': 'Spanish',
        'version': 'v0.0.1',
        'creator': 'Jaco-Assistant',
        'acoustic': f'{STT_HOST}/spanish/jaco-assistant/v0.0.1/model.tflite',
        'scorer': f'{STT_HOST}/spanish/jaco-assistant/v0.0.1/kenlm_es.scorer',
    }
    
}