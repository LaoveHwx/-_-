import time


class FPSCounter:
    """FPS counter that updates the displayed value at fixed intervals."""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._last_time = None
        self._display_fps = 0.0
        self._interval_start = None
        self._frame_count = 0

    def update(self) -> float:
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            self._interval_start = now
            return 0.0

        self._last_time = now
        self._frame_count += 1

        if self._interval_start is None:
            self._interval_start = now

        elapsed = now - self._interval_start
        if elapsed >= self.update_interval and elapsed > 0:
            self._display_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._interval_start = now

        return self._display_fps