"""
Stress test for the Record3D C++ library.

Exercises dangerous patterns to expose memory leaks, crashes, and race conditions:
  - Rapid connect/disconnect cycles
  - Reading frames without waiting for on_new_frame
  - Concurrent frame access from multiple threads
  - Double disconnect
  - Destroying session objects while streaming
  - Accessing accessors on a never-connected session
"""

import sys
import time
import threading
import traceback
from record3d import Record3DStream


def get_device():
    """Return the first connected device or None."""
    devs = Record3DStream.get_connected_devices()
    if not devs:
        print('ERROR: No device connected. Plug in an iPhone with Record3D running.')
        sys.exit(1)
    return devs[0]


def connect_with_retry(session, dev, max_attempts=5):
    """Try to connect, retrying if the device needs recovery time."""
    for attempt in range(max_attempts):
        if session.connect(dev):
            return True
        time.sleep(0.5)
    return False


def test_version():
    """Sanity: version string should be non-empty."""
    v = Record3DStream.get_version()
    assert isinstance(v, str) and len(v) > 0, f'Bad version: {v}'
    print(f'  [OK] get_version() = {v}')


def test_read_before_connect():
    """Access all frame accessors on a session that was never connected."""
    s = Record3DStream()
    # These should not crash — just return empty/zero data
    _ = s.get_depth_frame()
    _ = s.get_rgb_frame()
    _ = s.get_confidence_frame()
    _ = s.get_misc_data()
    _ = s.get_intrinsic_mat()
    _ = s.get_camera_pose()
    _ = s.get_device_type()
    del s
    print('  [OK] read_before_connect')


def test_double_disconnect():
    """Calling disconnect twice should be safe."""
    dev = get_device()
    s = Record3DStream()
    s.on_new_frame = lambda: None
    s.on_stream_stopped = lambda: None
    assert connect_with_retry(s, dev), 'Failed to connect'
    time.sleep(0.3)
    s.disconnect()
    s.disconnect()  # second disconnect — must not crash
    del s
    print('  [OK] double_disconnect')


def test_rapid_connect_disconnect(cycles=10):
    """Connect and immediately disconnect many times in a row."""
    dev = get_device()
    for i in range(cycles):
        s = Record3DStream()
        s.on_new_frame = lambda: None
        s.on_stream_stopped = lambda: None
        assert connect_with_retry(s, dev), f'Failed to connect on cycle {i}'
        # Disconnect almost immediately — sometimes before any frame arrives
        s.disconnect()
        del s
    print(f'  [OK] rapid_connect_disconnect x{cycles}')


def test_read_without_waiting(duration=2.0):
    """
    Connect and aggressively read frames without waiting for the on_new_frame
    callback — exercises the frame mutex under contention.
    """
    dev = get_device()
    s = Record3DStream()
    frame_count = [0]

    def on_frame():
        frame_count[0] += 1

    s.on_new_frame = on_frame
    s.on_stream_stopped = lambda: None
    assert connect_with_retry(s, dev), 'Failed to connect'

    end_time = time.time() + duration
    reads = 0
    while time.time() < end_time:
        _ = s.get_depth_frame()
        _ = s.get_rgb_frame()
        _ = s.get_confidence_frame()
        _ = s.get_misc_data()
        _ = s.get_intrinsic_mat()
        _ = s.get_camera_pose()
        _ = s.get_device_type()
        reads += 1

    s.disconnect()
    del s
    print(f'  [OK] read_without_waiting — {reads} read bursts, {frame_count[0]} frames received')


def test_concurrent_readers(duration=2.0, num_threads=4):
    """
    Multiple threads hammering all accessors simultaneously while stream is active.
    """
    dev = get_device()
    s = Record3DStream()
    frame_count = [0]
    stop_flag = threading.Event()

    def on_frame():
        frame_count[0] += 1

    s.on_new_frame = on_frame
    s.on_stream_stopped = lambda: None
    assert connect_with_retry(s, dev), 'Failed to connect'

    errors = []

    def reader(thread_id):
        count = 0
        try:
            while not stop_flag.is_set():
                _ = s.get_depth_frame()
                _ = s.get_rgb_frame()
                _ = s.get_confidence_frame()
                _ = s.get_misc_data()
                _ = s.get_intrinsic_mat()
                _ = s.get_camera_pose()
                _ = s.get_device_type()
                count += 1
        except Exception as e:
            errors.append((thread_id, e, traceback.format_exc()))

    threads = []
    for t in range(num_threads):
        th = threading.Thread(target=reader, args=(t,), daemon=True)
        th.start()
        threads.append(th)

    time.sleep(duration)
    stop_flag.set()
    for th in threads:
        th.join(timeout=5)

    s.disconnect()
    del s

    if errors:
        for tid, exc, tb in errors:
            print(f'  [FAIL] Thread {tid}: {exc}\n{tb}')
        raise RuntimeError('concurrent_readers had errors')

    print(f'  [OK] concurrent_readers — {num_threads} threads, {frame_count[0]} frames')


def test_destroy_while_streaming():
    """
    Let the session object get garbage-collected while the stream is still active.
    The destructor should cleanly shut everything down.
    """
    dev = get_device()
    s = Record3DStream()
    s.on_new_frame = lambda: None
    s.on_stream_stopped = lambda: None
    assert connect_with_retry(s, dev), 'Failed to connect'
    time.sleep(0.5)
    # Intentionally NOT calling disconnect — just delete
    del s
    # Force a GC pass
    import gc
    gc.collect()
    time.sleep(0.3)
    print('  [OK] destroy_while_streaming')


def test_reconnect_same_session():
    """Connect, disconnect, then reconnect the same session object."""
    dev = get_device()
    s = Record3DStream()
    s.on_new_frame = lambda: None
    s.on_stream_stopped = lambda: None

    for i in range(3):
        assert connect_with_retry(s, dev), f'Failed to connect on cycle {i}'
        time.sleep(0.3)
        s.disconnect()

    del s
    print('  [OK] reconnect_same_session x3')


def test_connect_while_connected():
    """Try to connect again while already connected — should return False, not crash."""
    dev = get_device()
    s = Record3DStream()
    s.on_new_frame = lambda: None
    s.on_stream_stopped = lambda: None

    assert connect_with_retry(s, dev), 'First connect should succeed'
    time.sleep(0.3)
    result2 = s.connect(dev)  # Should be rejected
    assert result2 == False, f'Second connect should be rejected, got {result2}'

    s.disconnect()
    del s
    print('  [OK] connect_while_connected')


def run_all_tests():
    tests = [
        test_version,
        test_read_before_connect,
        test_double_disconnect,
        test_connect_while_connected,
        test_rapid_connect_disconnect,
        test_read_without_waiting,
        test_concurrent_readers,
        test_destroy_while_streaming,
        test_reconnect_same_session,
    ]

    print(f'Running {len(tests)} stress tests...')
    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f'  [FAIL] {name}: {e}')
            traceback.print_exc()
            failed += 1
        # Small pause between tests to let the device settle
        time.sleep(0.5)

    print(f'\nResults: {passed} passed, {failed} failed out of {len(tests)} tests')
    return failed == 0


if __name__ == '__main__':
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
