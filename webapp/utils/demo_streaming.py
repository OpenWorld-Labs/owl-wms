import asyncio
import torch
import termcolor

from webapp.utils.render import generate_dummy_actions, save_video
from webapp.streaming import StreamingConfig, StreamingFrameGenerator
from webapp.utils.models import load_models
from webapp.utils.action_builder import ActionPattern
from webapp.utils.configs import SamplingConfig


async def demo_streaming_generation(pattern=ActionPattern.LOOK_AROUND):
    """Generate one batch using StreamingFrameGenerator instead of regular sampler."""
    
    print(termcolor.colored("🎮 OWL-WMS Streaming Demo", "green"))
    print(termcolor.colored("=" * 50, "green"))
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    streaming_config = StreamingConfig(
        fps=20,
        frames_per_batch=8,
        window_length=60,
        device=device
    )
    
    sampling_config = SamplingConfig()
    
    # Load models (reuse render.py's load_models)
    print("📦 Loading models...")
    encoder, decoder, train_config = load_models(device=device, verbose=True)
    model_config = train_config.model
    training_config = train_config.train
    
    # Create streaming frame generator
    print("🎬 Creating streaming frame generator...")
    frame_generator = StreamingFrameGenerator(
        encoder, decoder,
        streaming_config, model_config, training_config, sampling_config
    )
    
    # Generate actions (reuse render.py's generate_dummy_actions)
    print(f"🎯 Generating {pattern.value} actions...")
    mouse_batch, button_batch = generate_dummy_actions(pattern, streaming_config.window_length)
    
    # Generate frames using streaming generator
    print("🎨 Generating frames with streaming generator...")
    with frame_generator:
        frame_batch = await frame_generator.generate_frame_batch(mouse_batch, button_batch)
    
    print(f"Generated {frame_batch.shape[1]} frames with shape: {frame_batch.shape}")
    
    # Save video (reuse render.py's save_video)
    print("💾 Saving video...")
    output_path = save_video(frame_batch, f"streaming_demo_{pattern.value.lower()}", fps=streaming_config.fps)
    
    print(termcolor.colored(f"🎉 Demo complete! Video: {output_path}", "green"))
    return output_path


if __name__ == "__main__":
    # Try different patterns by changing this:
    pattern = ActionPattern.LOOK_AROUND  # or AIM_AND_SHOOT, CIRCLE_STRAFE, etc.
    
    print("Available patterns:")
    for p in ActionPattern:
        print(f"  - {p.value}")
    print()
    
    asyncio.run(demo_streaming_generation(pattern))