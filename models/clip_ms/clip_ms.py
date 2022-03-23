
from pyexpat import model
import sys
sys.path.insert(0, "/opt/program")
sys.path.insert(0, "/notebooks/nebula3_database")
sys.path.insert(0, "/notebooks")
from vlm.clip_api import CLIP_API

class Model(object):
    clip = CLIP_API('vit')
    @staticmethod
    def metadata():
        return {
            'signature_name': 'serving_default',
            'inputs': {
                'text': {
                    'dtype': 'str'
                },
                'movie_id': {
                    'dtype': 'str'
                },
                'scene_element': {
                    'dtype': 'int'
                },
            },
            'outputs': {
                'vector': {
                    'dtype': 'str'
                }
            }
        }

    @classmethod
    def predict(cls, data):
        models = cls.clip.clip_model.available_models()
        return ({'clip_data': data,
                  'clip_models': models   
                })

    @classmethod
    def encode_text(cls, data):
        #models = data
        vectors = cls.clip.clip_encode_text(data['text'])
        return ({'clip_data': data,
                  'clip_vector': vectors   
                })
    
    @classmethod
    def encode_video(cls, data):
        vectors = cls.clip.clip_encode_video(data['movie_id'], data['scene_element'])
        return ({'clip_data': data,
                  'clip_models': vectors   
                })
