import sys
import types
import warnings


def test_segment_continues_after_demo_failure(monkeypatch):
    import peg_analysis.segment as module

    class FakeFile:
        def __enter__(self): return object()
        def __exit__(self, *args): return False

    fake_h5py = types.SimpleNamespace(File=lambda *args, **kwargs: FakeFile())
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)
    monkeypatch.setattr(module, "selected_demos", lambda h, requested=None: ["bad", "good"])
    calls = []

    def fake_segment_demo(c, demo, overwrite=False, runner=None):
        calls.append(demo)
        if demo == "bad":
            raise ValueError("Expected one gap, found 0")
        return object()

    monkeypatch.setattr(module, "_segment_demo", fake_segment_demo)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module.segment({"dataset_path": "unused.h5"})

    assert calls == ["bad", "good"]
    assert len(caught) == 1
    assert "Skipping segmentation for bad" in str(caught[0].message)
