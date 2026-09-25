"""Bounded SSD read-ahead for sequential passes, exact reads for sparse passes."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import time
import shutil
import threading
import json


class ReadAheadCache:
    def __init__(self, directory, *, block_mb=256, slots=3, workers=4):
        if type(block_mb) is not int or not 1 <= block_mb <= 4096:
            raise ValueError('Cache block must be 1–4096 MiB')
        if type(slots) is not int or not 2 <= slots <= 16:
            raise ValueError('Cache slots must be 2–16')
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError('Download workers must be 1–16')
        parent = Path(directory).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free < block_mb*1024**2*slots + 64*1024**2:
            raise OSError('Insufficient free space for the bounded read cache')
        self.root = Path(tempfile.mkdtemp(prefix='terasort-read-', dir=parent)).resolve()
        self.block_bytes, self.slots, self.workers = block_mb*1024**2, slots, workers
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='terasort-download')
        self.files, self.future = OrderedDict(), None
        self.direct, self.direct_path = None, None
        self.counter = 0
        self.network_bytes = self.direct_bytes = self.cache_bytes = 0
        self.wait_seconds = 0.
        self.download_lock = threading.Lock()
        self.report_time = time.monotonic()
        self.report_bytes = 0

    def _record_download(self, count):
        with self.download_lock:
            self.network_bytes += count
            now = time.monotonic()
            elapsed = now-self.report_time
            if elapsed >= 5:
                delta = self.network_bytes-self.report_bytes
                try:
                    print('TERASORT_READ_CACHE_PROGRESS ' + json.dumps({
                        'downloaded_bytes': self.network_bytes,
                        'download_mib_s': round(delta/elapsed/1024**2, 1),
                        'workers': self.workers}), flush=True)
                except OSError:
                    pass
                self.report_time, self.report_bytes = now, self.network_bytes

    def _download(self, key, target):
        path, index = key
        before = path.stat()
        offset = index*self.block_bytes
        remaining = min(self.block_bytes, before.st_size-offset)
        if remaining <= 0:
            raise ValueError('Cache block outside source')
        partial = target.with_suffix('.partial')
        try:
            with partial.open('xb') as dest:
                dest.truncate(remaining)

            chunk_bytes = 8*1024**2
            worker_count = min(self.workers, max(1, (remaining+chunk_bytes-1)//chunk_bytes))
            segment_bytes = (remaining+worker_count-1)//worker_count

            def copy_range(start, stop):
                with path.open('rb', buffering=0) as source, partial.open('r+b', buffering=0) as dest:
                    source.seek(offset+start)
                    dest.seek(start)
                    todo = stop-start
                    while todo:
                        payload = source.read(min(chunk_bytes, todo))
                        if not payload:
                            raise OSError('Short network read')
                        view = memoryview(payload)
                        while view:
                            written = dest.write(view)
                            if not written:
                                raise OSError('Short local cache write')
                            view = view[written:]
                        todo -= len(payload)
                        self._record_download(len(payload))

            ranges = [(start, min(start+segment_bytes, remaining))
                      for start in range(0, remaining, segment_bytes)]
            if len(ranges) == 1:
                copy_range(*ranges[0])
            else:
                with ThreadPoolExecutor(max_workers=worker_count,
                                        thread_name_prefix='terasort-range') as workers:
                    futures = [workers.submit(copy_range, start, stop) for start, stop in ranges]
                    for future in futures:
                        future.result()
            after = path.stat()
            if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                raise OSError('Source changed while downloading cache block')
            partial.replace(target)
            return target
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    def _settle(self):
        if self.future is not None:
            key, future = self.future
            begin = time.perf_counter()
            try:
                self.files[key] = future.result()
            finally:
                self.wait_seconds += time.perf_counter()-begin
                self.future = None

    def _schedule(self, key, protect=None):
        if self.future is not None or key in self.files:
            return
        while len(self.files) >= self.slots:
            victim = next(k for k in self.files if k != protect)
            self.files.pop(victim).unlink()
        target = self.root / f'{self.counter:012d}.block'
        self.counter += 1
        self.future = (key,self.pool.submit(self._download,key,target))

    def read(self, path, start, stop, *, sequential=False):
        path = Path(path).resolve()
        # Kilosort's concatenated-file offsets may be NumPy integer scalars.
        start, stop = int(start), int(stop)
        if not 0 <= start < stop:
            raise ValueError('Invalid byte range')
        if self.future is not None and self.future[1].done():
            self._settle()
        if not sequential:
            # Strided Kilosort calibration must not download the unused gaps.
            if self.direct_path != path:
                if self.direct is not None:
                    self.direct.close()
                self.direct = path.open('rb',buffering=0)
                self.direct_path = path
            self.direct.seek(start)
            payload = self.direct.read(stop-start)
            self.direct_bytes += len(payload)
            if len(payload) != stop-start:
                raise OSError('Short direct source read')
            return payload
        output = bytearray(stop-start)
        position = start
        while position < stop:
            index = position//self.block_bytes
            key = (path,index)
            if key not in self.files:
                self._settle()
                if key not in self.files:
                    self._schedule(key)
                    self._settle()
            local = self.files[key]
            self.files.move_to_end(key)
            count = min(stop-position,local.stat().st_size-position%self.block_bytes)
            if count <= 0:
                raise OSError('Source range exceeds cached file')
            with local.open('rb') as handle:
                handle.seek(position%self.block_bytes)
                payload = handle.read(count)
            if len(payload) != count:
                raise OSError('Incomplete local cache read')
            output[position-start:position-start+count] = payload
            self.cache_bytes += count
            position += count
            if (index+1)*self.block_bytes < path.stat().st_size:
                self._schedule((path,index+1),protect=key)
        return output

    def stats(self):
        return dict(downloaded_bytes=self.network_bytes,direct_bytes=self.direct_bytes,
                    cache_served_bytes=self.cache_bytes,wait_seconds=self.wait_seconds,
                    disk_budget_bytes=self.block_bytes*self.slots,
                    read_buffer_bytes=8*1024**2,download_workers=self.workers)

    def close(self):
        if self.direct is not None:
            self.direct.close()
        self.pool.shutdown(wait=True,cancel_futures=True)
        # Only delete files created inside this instance's unique cache directory.
        for path in self.root.iterdir():
            if path.suffix in ('.block','.partial') and path.stem.isdigit():
                path.unlink()
        self.root.rmdir()
