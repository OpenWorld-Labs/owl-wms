from .gamerft import GameRFT, GameRFTCore

def get_model_cls(model_id):
    if model_id == "game_rft":
        return GameRFT
    if model_id == "game_rft_core":
        return GameRFTCore




