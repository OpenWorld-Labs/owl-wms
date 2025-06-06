import cv2
import time
import json
import torch
import base64
import asyncio
import termcolor
import numpy as np

from fastapi import WebSocket
from webapp.action_converter import ActionCollector
from webapp.streaming import StreamingFrameGenerator, FrameBuffer

class UserGameSession:
    """Main orchestrator for single-user real-time gameplay."""
    def __init__(self, frame_generator: StreamingFrameGenerator):
        self.frame_generator    = frame_generator
        self.action_collector   = ActionCollector(frame_generator.streaming_config)
        self.frame_buffer       = FrameBuffer(frame_generator.streaming_config)
        
    async def run_session(self, websocket):
        with self.frame_generator:
            print(termcolor.colored(f"Starting streaming session at {self.frame_generator.streaming_config.fps} FPS", "green"))
            print(termcolor.colored(f"Generating                    {self.frame_generator.streaming_config.frames_per_batch} frames per batch", "green"))
            print(termcolor.colored(f"Batch duration:               {self.frame_generator.streaming_config.batch_duration:.3f}s", "green"))
            
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._action_input_loop(websocket))
                tg.create_task(self._frame_generation_loop())
                tg.create_task(self._frame_output_loop(websocket))
    
    async def _action_input_loop(self, websocket):
        async for message in websocket:
            try:
                action_data = json.loads(message)
                await self.action_collector.add_websocket_action(action_data)
            except Exception as e:
                print(f"Error processing action: {e}")

    async def _frame_generation_loop(self):
        """Generate frame batches continuously."""
        print(termcolor.colored("Frame generation loop started", "green"))
        while True:
            try:
                # Collect 8 frames worth of actions
                mouse_batch, button_batch = await self.action_collector.collect_batch()
                # Generate 8 frames
                frame_batch = await self.frame_generator.generate_frame_batch(mouse_batch, button_batch)
                # Queue frames for streaming
                await self.frame_buffer.add_frame_batch(frame_batch)
            except Exception as e:
                import traceback
                print(termcolor.colored(f"Error in frame generation: {e} :\n {traceback.format_exc()}", "red"))
                await asyncio.sleep(0.1)  # Brief pause before retry
    
    async def _frame_output_loop(self, websocket):
        while True:
            try:
                frame = await self.frame_buffer.get_next_frame()
                await self._send_frame_to_client(websocket, frame)
            except Exception as e:
                print(termcolor.colored(f"Error in frame streaming: {e}", "red"))
                await asyncio.sleep(self.frame_generator.streaming_config.frame_interval)
    
    async def _send_frame_to_client(self, websocket: WebSocket, frame: torch.Tensor):
        # Convert frame to base64 JPEG
        print(f"Frame shape: {frame.shape} - with stats: {frame.min()=}, {frame.max()=}, {frame.mean()=}, {frame.std()=}")
        frame_np = frame.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        frame_np = ((frame_np + 1) * 127.5).clip(0, 255).astype(np.uint8)  # Normalize
        
        _, buffer = cv2.imencode('.jpg', frame_np)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        await websocket.send(json.dumps({
            "type": "frame",
            "data": frame_base64,
            "timestamp": time.time()
        }))
