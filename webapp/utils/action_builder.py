import math
import torch
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple, Callable


# Button mapping from the codebase
BUTTON_NAMES = ["W", "A", "S", "D", "LSHIFT", "SPACE", "R", "F", "E", "LMB", "RMB"]
BUTTON_INDICES = {name: idx for idx, name in enumerate(BUTTON_NAMES)}


class ActionPattern(Enum):
    IDLE            = "idle"
    WALK_FORWARD    = "walk_forward"
    STRAFE_LEFT     = "strafe_left"
    STRAFE_RIGHT    = "strafe_right"
    WALK_BACKWARD   = "walk_backward"
    CIRCLE_STRAFE   = "circle_strafe"
    LOOK_AROUND     = "look_around"
    AIM_AND_SHOOT   = "aim_and_shoot"
    SPRINT_FORWARD  = "sprint_forward"
    RELOAD          = "reload"


@dataclass(frozen=True)
class ActionConfig:
    sequence_length: int
    device: Union[str, torch.device] = 'cpu'
    dtype: torch.dtype = torch.float32
    n_buttons: int = 11
    mouse_range: Tuple[float, float] = (-1.0, 1.0)
    smooth_transitions: bool = True
    random_seed: Optional[int] = None


class MouseGenerator:
    @staticmethod
    def _apply_smoothing(window_size: int, values: torch.Tensor) -> torch.Tensor:
        """Apply smoothing to mouse movements using convolution."""
        if window_size <= 1:
            return values
        
        # values is [2, sequence_length] for mouse x,y coordinates
        # Use groups=2 to smooth each channel independently
        kernel = torch.ones(2, 1, window_size, device=values.device) / window_size
        padding = window_size // 2
        padded = torch.nn.functional.pad(values, (padding, padding), mode='reflect')
        smoothed = torch.nn.functional.conv1d(
            padded.unsqueeze(0),  # Add batch dim: [1, 2, sequence_length]
            kernel, padding=0,
            groups=2              # Each output channel only depends on corresponding input channel
        ).squeeze(0)              # Remove batch dim: [2, sequence_length]
        return smoothed
    
    @staticmethod
    def idle(config: ActionConfig) -> torch.Tensor:
        """Generate idle mouse movement (minimal random drift)."""
        return torch.randn(config.sequence_length, 2, 
                          device=config.device, dtype=config.dtype) * 0.05
    
    @staticmethod
    def look_around(config: ActionConfig, 
                   speed: float = 0.3, 
                   amplitude: float = 0.7) -> torch.Tensor:
        """Generate smooth looking around movement."""
        t = torch.linspace(0, 4 * math.pi, config.sequence_length, 
                          device=config.device, dtype=config.dtype)
        
        # Create smooth sinusoidal movement
        mouse_x = amplitude * torch.sin(t * speed) * torch.cos(t * speed * 0.3)
        mouse_y = amplitude * torch.cos(t * speed * 0.7) * torch.sin(t * speed * 0.2)
        
        movement = torch.stack([mouse_x, mouse_y], dim=1)
        
        if config.smooth_transitions:
            return MouseGenerator._apply_smoothing(5, movement.T).T
        
        return movement
    
    @staticmethod
    def aim_tracking(config: ActionConfig,
                    target_speed: float = 0.1,
                    noise_level: float = 0.02) -> torch.Tensor:
        """Generate aiming/tracking movement with micro-adjustments."""
        # Create base tracking movement
        t = torch.linspace(0, 2 * math.pi, config.sequence_length,
                          device=config.device, dtype=config.dtype)
        
        # Smooth circular tracking
        base_x = 0.3 * torch.sin(t * target_speed)
        base_y = 0.2 * torch.cos(t * target_speed * 1.2)
        
        # Add realistic micro-movements
        noise_x = torch.randn(config.sequence_length, device=config.device) * noise_level
        noise_y = torch.randn(config.sequence_length, device=config.device) * noise_level
        
        return torch.stack([base_x + noise_x, base_y + noise_y], dim=1)
    
    @staticmethod
    def custom_path(config: ActionConfig, 
                   path_points: List[Tuple[float, float]],
                   interpolation: str = 'linear') -> torch.Tensor:
        """Generate mouse movement following a custom path."""
        if len(path_points) < 2:
            return MouseGenerator.idle(config)
        
        # Convert to tensors
        points = torch.tensor(path_points, device=config.device, dtype=config.dtype)
        
        # Create interpolation indices
        t = torch.linspace(0, len(points) - 1, config.sequence_length,
                          device=config.device, dtype=config.dtype)
        
        # Linear interpolation between points
        indices = t.long()
        weights = t - indices.float()
        
        # Handle edge case
        indices = torch.clamp(indices, 0, len(points) - 2)
        weights = weights.unsqueeze(1)
        
        interpolated = (1 - weights) * points[indices] + weights * points[indices + 1]
        
        return interpolated


class ButtonGenerator:
    @staticmethod
    def idle(config: ActionConfig) -> torch.Tensor:
        """Generate idle button state (all buttons released)."""
        return torch.zeros(config.sequence_length, config.n_buttons,
                          device=config.device, dtype=config.dtype)
    
    @staticmethod
    def hold_buttons(config: ActionConfig, 
                    button_names: List[str],
                    start_frame: int = 0,
                    duration: Optional[int] = None) -> torch.Tensor:
        """Hold specific buttons for a duration."""
        buttons = torch.zeros(config.sequence_length, config.n_buttons,
                             device=config.device, dtype=config.dtype)
        
        end_frame = start_frame + (duration or config.sequence_length)
        end_frame = min(end_frame, config.sequence_length)
        
        for button_name in button_names:
            if button_name in BUTTON_INDICES:
                idx = BUTTON_INDICES[button_name]
                buttons[start_frame:end_frame, idx] = 1.0
        
        return buttons
    
    @staticmethod
    def tap_sequence(config: ActionConfig,
                    button_sequences: List[Tuple[str, int, int]]) -> torch.Tensor:
        """Create button taps at specific times.
        
        Args:
            button_sequences: List of (button_name, start_frame, duration) tuples
        """
        buttons = torch.zeros(config.sequence_length, config.n_buttons,
                             device=config.device, dtype=config.dtype)
        
        for button_name, start_frame, duration in button_sequences:
            if button_name in BUTTON_INDICES and start_frame < config.sequence_length:
                idx = BUTTON_INDICES[button_name]
                end_frame = min(start_frame + duration, config.sequence_length)
                buttons[start_frame:end_frame, idx] = 1.0
        
        return buttons
    
    @staticmethod
    def pattern_from_name(config: ActionConfig, pattern: ActionPattern) -> torch.Tensor:
        """Generate button pattern from predefined patterns."""
        if pattern == ActionPattern.WALK_FORWARD:
            return ButtonGenerator.hold_buttons(config, ["W"])
        elif pattern == ActionPattern.STRAFE_LEFT:
            return ButtonGenerator.hold_buttons(config, ["A"])
        elif pattern == ActionPattern.STRAFE_RIGHT:
            return ButtonGenerator.hold_buttons(config, ["D"])
        elif pattern == ActionPattern.WALK_BACKWARD:
            return ButtonGenerator.hold_buttons(config, ["S"])
        elif pattern == ActionPattern.SPRINT_FORWARD:
            return ButtonGenerator.hold_buttons(config, ["W", "LSHIFT"])
        elif pattern == ActionPattern.RELOAD:
            return ButtonGenerator.tap_sequence(config, [("R", 10, 20)])
        else:
            return ButtonGenerator.idle(config)


class ActionSequenceBuilder:
    def __init__(self, config: ActionConfig):
        self.config = config
        self.mouse_sequence = torch.zeros(config.sequence_length, 2,
                                        device=config.device, dtype=config.dtype)
        self.button_sequence = torch.zeros(config.sequence_length, config.n_buttons,
                                         device=config.device, dtype=config.dtype)
    
    def add_mouse_segment(self, 
                         start_frame: int, 
                         end_frame: int,
                         generator_func: Callable[[ActionConfig], torch.Tensor],
                         **kwargs) -> 'ActionSequenceBuilder':
        """Add a mouse movement segment."""
        segment_config = ActionConfig(
            sequence_length=end_frame - start_frame,
            device=self.config.device,
            dtype=self.config.dtype,
            n_buttons=self.config.n_buttons
        )
        
        segment = generator_func(segment_config, **kwargs)
        self.mouse_sequence[start_frame:end_frame] = segment
        return self
    
    def add_button_segment(self,
                          start_frame: int,
                          end_frame: int,
                          generator_func: Callable[[ActionConfig], torch.Tensor],
                          **kwargs) -> 'ActionSequenceBuilder':
        """Add a button press segment."""
        segment_config = ActionConfig(
            sequence_length=end_frame - start_frame,
            device=self.config.device,
            dtype=self.config.dtype,
            n_buttons=self.config.n_buttons
        )
        
        segment = generator_func(segment_config, **kwargs)
        self.button_sequence[start_frame:end_frame] = segment
        return self
    
    def build(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build and return the final action sequence."""
        return self.mouse_sequence, self.button_sequence


class ActionSequenceGenerator:
    def __init__(self, config: ActionConfig):
        self.config = config
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)
    
    def generate_pattern(self, 
                        pattern: ActionPattern,
                        mouse_kwargs: Optional[Dict] = None,
                        button_kwargs: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate actions for a predefined pattern."""
        mouse_kwargs = mouse_kwargs or {}
        button_kwargs = button_kwargs or {}
        
        if pattern == ActionPattern.IDLE:
            mouse = MouseGenerator.idle(self.config)
            buttons = ButtonGenerator.idle(self.config)
        
        elif pattern == ActionPattern.LOOK_AROUND:
            mouse = MouseGenerator.look_around(self.config, **mouse_kwargs)
            buttons = ButtonGenerator.idle(self.config)
        
        elif pattern == ActionPattern.AIM_AND_SHOOT:
            mouse = MouseGenerator.aim_tracking(self.config, **mouse_kwargs)
            # Add some shooting
            shoot_times = [(i * 30, 5) for i in range(self.config.sequence_length // 30)]
            buttons = ButtonGenerator.tap_sequence(
                self.config, 
                [("LMB", start, dur) for start, dur in shoot_times]
            )
        
        elif pattern == ActionPattern.CIRCLE_STRAFE:
            # Combine circular mouse movement with strafing
            mouse = MouseGenerator.look_around(self.config, speed=0.2, amplitude=0.5)
            buttons = ButtonGenerator.hold_buttons(self.config, ["A", "W"])
        
        else:
            mouse = MouseGenerator.idle(self.config)
            buttons = ButtonGenerator.pattern_from_name(self.config, pattern)
        
        return mouse, buttons
    
    def generate_custom_sequence(self) -> ActionSequenceBuilder:
        """Get a builder for creating custom action sequences."""
        return ActionSequenceBuilder(self.config)
    
    def generate_batch(self, 
                      batch_size: int,
                      patterns: Optional[List[ActionPattern]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of action sequences."""
        if patterns is None:
            patterns = [ActionPattern.IDLE] * batch_size
        
        if len(patterns) != batch_size:
            # Repeat or truncate patterns to match batch size
            patterns = (patterns * (batch_size // len(patterns) + 1))[:batch_size]
        
        mouse_batch = []
        button_batch = []
        
        for pattern in patterns:
            mouse, buttons = self.generate_pattern(pattern)
            mouse_batch.append(mouse)
            button_batch.append(buttons)
        
        return torch.stack(mouse_batch), torch.stack(button_batch)


if __name__ == "__main__":
    print("=== Action Generation Examples ===")
    config = ActionConfig(sequence_length=100)
    builder = ActionSequenceGenerator(config).generate_custom_sequence()
    
    mouse_custom, button_custom = (builder
                                  .add_mouse_segment(0, 30, MouseGenerator.idle)
                                  .add_mouse_segment(30, 70, MouseGenerator.look_around, speed=0.5)
                                  .add_mouse_segment(70, 100, MouseGenerator.aim_tracking)
                                  .add_button_segment(0, 50, ButtonGenerator.hold_buttons, button_names=["W"])
                                  .add_button_segment(50, 100, ButtonGenerator.hold_buttons, button_names=["W", "LSHIFT"])
                                  .build())

    print(f"Custom sequence mouse shape: {mouse_custom.shape}, button shape: {button_custom.shape}")
    print("Button names mapping:", BUTTON_INDICES)