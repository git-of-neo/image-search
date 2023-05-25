from contextlib import asynccontextmanager
import vit_jax.models_mixer as model
import ml_collections
import numpy as np
from PIL import Image
from vit_jax.checkpoint import load
from fastapi import FastAPI, UploadFile
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from uuid import uuid4, UUID

## Model
config = ml_collections.ConfigDict()
# Mixer B_16 config from https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py
config.model_name = "Mixer-B_16"
config.patches = ml_collections.ConfigDict({"size": (16, 16)})
config.hidden_dim = 768
config.num_blocks = 12
config.tokens_mlp_dim = 384
config.channels_mlp_dim = 3072

vector_size = 21843
mixer = model.MlpMixer(num_classes=vector_size, **config)  # 21k for ImageNet21k

# inference following guide : https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb#scrollTo=TE7BoGkyoY7X
params = load("imagenet21k_Mixer-B_16.npz")
params["pre_logits"] = {}


def preprocess(image: Image.Image) -> Image.Image:
    # resize then centre crop to 224x224 per appendix of https://arxiv.org/abs/2105.01601
    return image.resize((256, 256)).crop((16, 16, 240, 240))


def vectorize(image: Image.Image):
    image = preprocess(image)
    (logits,) = mixer.apply(
        dict(params=params), (np.array(image) / 128 - 1)[None, ...], train=False
    )
    return logits.tolist()


## db setup
client = QdrantClient(path="qdrant.db")
try:
    client.get_collection("images")
    db_ready = True
except ValueError:
    client.recreate_collection(
        collection_name="images",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    db_ready = False


def _insert_image(img: Image.Image, image_id: UUID):
    client.upsert(
        "images", points=[PointStruct(id=str(image_id), vector=vectorize(img))]
    )


def _save_static_image(img: Image.Image, idx: UUID):
    img.save("stored/" + str(idx) + ".jpeg", format="jpeg")


## server setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not db_ready:
        import os

        def is_jpg(s: str):
            return s[-4:] == ".jpg"

        static_dir = "static/"
        for fname in filter(is_jpg, os.listdir(static_dir)):
            _insert_image(Image.open(static_dir + fname), uuid4())

    yield


app = FastAPI(lifespan=lifespan)


## endpoints
@app.post("/images")
async def insert_image(uploaded_file: UploadFile):
    image = Image.open(BytesIO(await uploaded_file.read()))
    idx = uuid4()
    _save_static_image(image, idx)
    _insert_image(image, idx)
    return {"message": "ok"}


@app.post("/images/search")
async def find_similar_images(uploaded_file: UploadFile):
    image = Image.open(BytesIO(await uploaded_file.read()))
    res = client.search(
        collection_name="images", query_vector=vectorize(image), limit=5
    )
    # TODO :  CDN for match images
    return {"urls": [r.id for r in res]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
