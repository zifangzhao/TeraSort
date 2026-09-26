from pathlib import Path
from threading import Event
from types import SimpleNamespace
import numpy as np
import pytest
from terasort.read_cache import ReadAheadCache
from terasort.kilosort_int16 import source_slice


def test_cache_exact_ranges_eviction_and_sparse_bypass(tmp_path):
    payload = np.random.default_rng(4).integers(0,256,3*1024**2+131,dtype=np.uint8).tobytes()
    source=tmp_path/'source.bin'; source.write_bytes(payload)
    cache=ReadAheadCache(tmp_path/'scratch',block_mb=1,slots=2)
    root=cache.root
    try:
        assert cache.read(source,71,99)==payload[71:99]
        assert not list(root.iterdir())
        assert cache.direct_bytes==28 and cache.network_bytes==0
        for start,stop in [(0,500),(1024**2-11,1024**2+111),(2*1024**2+7,len(payload)),(71,99)]:
            assert cache.read(source,start,stop,sequential=True)==payload[start:stop]
            cache._settle()
            assert sum(p.stat().st_size for p in root.iterdir()) <= 2*1024**2
            assert len(list(root.iterdir()))<=2
        with pytest.raises(OSError): cache.read(source,len(payload),len(payload)+1)
    finally:
        cache.close()
    assert not root.exists()
    assert source.read_bytes()==payload


def test_download_overlaps_consumption(tmp_path,monkeypatch):
    source=tmp_path/'source.bin'; source.write_bytes(bytes(3*1024**2))
    cache=ReadAheadCache(tmp_path/'scratch',block_mb=1,slots=2)
    started,release=Event(),Event()
    original=cache._download
    def delayed(key,target):
        if key[1]==1:
            started.set()
            if not release.wait(5): raise RuntimeError('Test timed out')
        return original(key,target)
    monkeypatch.setattr(cache,'_download',delayed)
    try:
        assert cache.read(source,0,20,sequential=True)==bytes(20)
        assert started.wait(2)
        assert not cache.future[1].done()
        assert cache.read(source,20,40,sequential=True)==bytes(20)
    finally:
        release.set()
        cache.close()


def test_adapter_preserves_int16_samples(tmp_path):
    raw=np.arange(5000*3,dtype=np.int16).reshape(-1,3)
    path=tmp_path/'input.bin'; raw.tofile(path)
    reader=SimpleNamespace(file=raw,filename=str(path),n_chan_bin=3)
    cache=ReadAheadCache(tmp_path/'scratch',block_mb=1,slots=2)
    try:
        for sequential in (False,True):
            np.testing.assert_array_equal(source_slice(reader,7,411,cache,sequential),raw[7:411])
    finally: cache.close()


def test_download_failure_never_publishes_partial(tmp_path):
    cache=ReadAheadCache(tmp_path/'scratch',block_mb=1,slots=2)
    try:
        with pytest.raises(FileNotFoundError):
            cache.read(tmp_path/'absent.bin',0,10,sequential=True)
        assert not list(cache.root.glob('*.block'))
    finally: cache.close()


def test_sparse_whitening_prefetch_preserves_exact_ranges_and_stays_bounded(tmp_path):
    payload = np.random.default_rng(123).integers(0, 256, 2*1024**2,
                                                   dtype=np.uint8).tobytes()
    source = tmp_path / 'source.bin'
    source.write_bytes(payload)
    cache = ReadAheadCache(tmp_path / 'scratch', block_mb=1, slots=2,
                           workers=4)
    try:
        stride, length = 128*1024, 32*1024
        for index in range(12):
            start = index*stride
            assert cache.read(source, start, start+length) == payload[start:start+length]
            assert len(cache.sparse_futures) <= cache.sparse_depth
        assert cache.sparse_served_bytes >= length
        assert cache.stats()['sparse_prefetch_budget_bytes'] == 256*1024**2
        # A change of access pattern cancels speculative sparse requests.
        assert cache.read(source, 17, 50) == payload[17:50]
        assert not cache.sparse_futures
        assert cache.read(source, 0, 100, sequential=True) == payload[:100]
    finally:
        cache.close()
    assert source.read_bytes() == payload
