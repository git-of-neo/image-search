## Similar image search using Qdrant and MlpMixer
This project uses a modified subset of the [vision transformer](https://github.com/google-research/vision_transformer) repository for 
creating image embeddings. See `vit_jax/README.md` for more details on the modifications made. 

## Getting Started
### Requirements
Install the requirements
```
python3 -r requirements.txt
```

This will not work on windows per jax's limitations.

Obtain a copy of the MlpMixer-B_16 checkpoint from the [official google cloud storage](https://console.cloud.google.com/storage/browser/mixer_models).

### Running the server
`python3 main.py`