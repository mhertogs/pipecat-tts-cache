import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import pytest
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from pipecat_tts_cache.backends.base import CacheBackend
from pipecat_tts_cache.mixin import TTSCacheMixin
from pipecat_tts_cache.models import CachedAudioChunk, CachedTTSResponse


# ---------------------------------------------------------------------------
# In-memory cache backend for tests
# ---------------------------------------------------------------------------
class DictCacheBackend(CacheBackend):
    def __init__(self):
        self._store: Dict[str, CachedTTSResponse] = {}

    async def get(self, key: str) -> Optional[CachedTTSResponse]:
        return self._store.get(key)

    async def set(self, key: str, response: CachedTTSResponse, ttl: Optional[int] = None) -> bool:
        self._store[key] = response
        return True

    async def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    async def clear(self, namespace: Optional[str] = None) -> int:
        n = len(self._store)
        self._store.clear()
        return n

    async def exists(self, key: str) -> bool:
        return key in self._store

    async def get_stats(self) -> Dict[str, Any]:
        return {"size": len(self._store)}

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Minimal fake TTS base that the mixin can inherit from.
# ---------------------------------------------------------------------------
class FakeTTSBase:
    def __init__(self, *args, **kwargs):
        self._voice_id = "test-voice"
        self.model_name = "test-model"
        self.sample_rate = 16000
        self._settings = {}
        self._pushed_frames: List[Frame] = []

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        yield TTSStartedFrame()
        yield TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
        yield TTSStoppedFrame()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self._pushed_frames.append(frame)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_websocket_ordering_miss_then_hit():
    """Key test: sentence 1 (cache miss, websocket) then sentence 2 (cache hit).

    Without the event gate, sentence 2's cached frames would arrive before
    sentence 1's background audio. The event gate ensures run_tts for sentence 1
    blocks until its audio is fully delivered, so sentence 2 waits its turn.

    This test tracks BOTH yielded frames (from the async generator) AND pushed
    frames (from background tasks via push_frame) in a single ordered list,
    which is necessary to detect the race condition.
    """
    # Shared ordered list of all downstream frames
    downstream: List[Tuple[str, Frame]] = []

    class SlowWebSocketTTSBase(FakeTTSBase):
        """Websocket TTS where run_tts yields only TTSStartedFrame.

        Audio + stopped frames are delivered later via push_frame from a
        background task with a deliberate delay to expose ordering bugs.
        """

        async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
            yield TTSStartedFrame()
            asyncio.create_task(self._deliver_audio_in_background())

        async def _deliver_audio_in_background(self):
            # Wait long time to ensure race condition is hit
            await asyncio.sleep(3)
            await self.push_frame(
                TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
            )
            await self.push_frame(TTSStoppedFrame())

        async def push_frame(
            self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
        ):
            downstream.append(("pushed", frame))

    class TestCachedTTS(TTSCacheMixin, SlowWebSocketTTSBase):
        pass

    backend = DictCacheBackend()
    tts = TestCachedTTS(cache_backend=backend)

    # Pre-populate cache for sentence 2
    cached_response = CachedTTSResponse(
        audio_chunks=[CachedAudioChunk(audio=b"\xff" * 320, sample_rate=16000, num_channels=1)],
        sample_rate=16000,
        num_channels=1,
        total_duration_s=0.01,
    )
    sentence2_key = tts._generate_cache_key("How are you?")
    await backend.set(sentence2_key, cached_response)

    # Sentence 1: cache miss (websocket-style, audio delivered via background push_frame)
    async for frame in tts.run_tts("Hello world"):
        downstream.append(("s1_yield", frame))

    # Yield to event loop so background task can start
    await asyncio.sleep(0)

    # Sentence 2: cache hit (inline cached frames)
    async for frame in tts.run_tts("How are you?"):
        downstream.append(("s2_yield", frame))

    # Wait for background task to finish delivering audio
    await asyncio.sleep(4.0)

    # Find indices for each sentence's frames
    s1_indices = [
        i for i, (tag, _) in enumerate(downstream) if tag.startswith("s1") or tag == "pushed"
    ]
    s2_indices = [i for i, (tag, _) in enumerate(downstream) if tag.startswith("s2")]

    # More precise: identify s1 frames as s1_yield frames + pushed frames that
    # belong to s1 (pushed before any s2 frame, or after all s2 frames if race exists)
    # Since pushed frames come from s1's background task, tag them as s1
    s1_indices = [i for i, (tag, _) in enumerate(downstream) if tag in ("s1_yield", "pushed")]
    s2_indices = [i for i, (tag, _) in enumerate(downstream) if tag.startswith("s2")]

    assert s1_indices, "Expected sentence 1 frames"
    assert s2_indices, "Expected sentence 2 frames"

    last_s1_idx = max(s1_indices)
    first_s2_idx = min(s2_indices)

    frame_log = [(tag, type(f).__name__) for tag, f in downstream]
    assert last_s1_idx < first_s2_idx, (
        f"Sentence 1 frames must all complete before sentence 2 starts.\n"
        f"  last_s1={last_s1_idx}, first_s2={first_s2_idx}\n"
        f"  downstream={frame_log}"
    )


@pytest.mark.asyncio
async def test_interruption_unblocks_waiting_run_tts():
    """An interruption should unblock a run_tts that's waiting for TTSStoppedFrame."""
    backend = DictCacheBackend()

    class SlowWebSocketTTSBase(FakeTTSBase):
        """Websocket TTS that never delivers audio (simulates hang)."""

        async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
            yield TTSStartedFrame()
            # No background task — audio never arrives

    class SlowCachedTTS(TTSCacheMixin, SlowWebSocketTTSBase):
        pass

    tts = SlowCachedTTS(cache_backend=backend)

    async def run_tts_with_interruption():
        # Start run_tts; it will block waiting for TTSStoppedFrame
        gen = tts.run_tts("Hello world")

        # Exhaust the generator's yielded frames
        frames = []
        async for frame in gen:
            frames.append(frame)

            # After getting the first frame, simulate an interruption
            if isinstance(frame, TTSStartedFrame):
                # Schedule the interruption to fire shortly
                asyncio.get_event_loop().call_soon(
                    asyncio.ensure_future,
                    tts._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM),
                )

        return frames

    # Should complete without timing out (the 30s timeout in run_tts
    # would be hit without the interruption unblocking)
    frames = await asyncio.wait_for(run_tts_with_interruption(), timeout=5.0)
    assert any(isinstance(f, TTSStartedFrame) for f in frames)
