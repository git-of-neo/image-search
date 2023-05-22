import vit_jax.models_mixer as model
import ml_collections 
import numpy as np
from PIL import Image
import collections
from vit_jax.checkpoint import load

config = ml_collections.ConfigDict()
# Mixer B_16 config from https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
config.model_name = 'Mixer-B_16'
config.patches = ml_collections.ConfigDict({'size': (16, 16)})
config.hidden_dim = 768
config.num_blocks = 12
config.tokens_mlp_dim = 384
config.channels_mlp_dim = 3072

mixer = model.MlpMixer(num_classes=21843, **config) # 21k for ImageNet21k

# inference following guide : https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb#scrollTo=TE7BoGkyoY7X
params = load("imagenet21k_Mixer-B_16.npz")
params['pre_logits'] = {}
resolution = 224
img = Image.open('car.jpg')

def vectorize(image : Image):
    logits, = mixer.apply(dict(params=params), (np.array(img) / 128 - 1)[None, ...], train=False)
    return logits
