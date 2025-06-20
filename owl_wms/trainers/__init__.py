def get_trainer_cls(trainer_id):
    if trainer_id == "rft":
        from .gamerft_trainer import RFTTrainer
        return RFTTrainer
    if trainer_id == "causvid":
        from .causvid import CausVidTrainer
        return CausVidTrainer
    if trainer_id == "av":
        from .av_trainer import AVRFTTrainer
        return AVRFTTrainer
    if trainer_id == "ddpo":
        from .ddpo_trainer import DDPOTrainer
        return DDPOTrainer