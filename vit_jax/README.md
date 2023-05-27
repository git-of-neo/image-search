## Modified vit_jax
This directory contains files (some modified) from [vision_transformer](https://github.com/google-research/vision_transformer) that are required to perform inference using MlpMixer. 

### Modifications
* checkpoint.py 
  * Stip file to bare minimum required to run a `load()` for the MlpMixer model
  * Modified `load()` to load from local checkpoint instead
* models_mixer.py
  * unmodified