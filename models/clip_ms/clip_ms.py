
import sys
import numpy as np
from pathlib import Path
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
                'image_data': {
                    'dtype': 'float'
                }
            },
            'outputs': {
                'class_probabilities': {
                    'dtype': 'float'
                }
            }
        }

    @classmethod
    def predict(cls, data):
        models = cls.clip.available_models()
        return ({'clip_data': data,
                  'clip_models': models   
                })
