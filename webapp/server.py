import os
from torch import nn

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from webapp.utils.models    import load_models
from webapp.streaming       import StreamingFrameGenerator
from webapp.user_session    import UserGameSession
from webapp.utils.configs   import WebappConfig


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# -- lifespan
encoder: nn.Module      = None
decoder: nn.Module      = None
config: WebappConfig    = None
webapp_config_path      = "../configs/webapp/config.yaml" ; assert os.path.exists(webapp_config_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, decoder, config
    with open(webapp_config_path, "r") as f:
        config: WebappConfig = WebappConfig.from_yaml(f)

    encoder, decoder, _ = load_models(
        checkpoint_path=config.model_checkpoint_path,
        config_path=config.model_config_path,
        device=config.device, verbose=True,
    )

    yield
    encoder, decoder, config = None, None, None


@app.websocket("/ws/game")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create streaming session for this user
    frame_generator = StreamingFrameGenerator(encoder, decoder,
                                              streaming_config=config.stream_config,
                                              model_config=config.model_config,
                                              train_config=config.run_config,
                                              sampling_config=config.sampling_config)
    session = UserGameSession(frame_generator)
    
    # Run the session (your existing code!)
    await session.run_session(websocket)
