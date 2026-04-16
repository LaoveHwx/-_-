import time


class FPSCounter:
    """Simple smoothed FPS counter for real-time loops."""

    def __init__(self, smooth_factor: float = 0.9):
        self.smooth_factor = smooth_factor
        self._last_time = None
        self._fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            return 0.0

        dt = now - self._last_time
        self._last_time = now
        if dt <= 0:
            return self._fps

        current_fps = 1.0 / dt
        if self._fps == 0.0:
            self._fps = current_fps
        else:
            a = self.smooth_factor
            self._fps = a * self._fps + (1.0 - a) * current_fps
        return self._fps