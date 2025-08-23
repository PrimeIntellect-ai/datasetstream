import asyncio
from typing import AsyncIterator, Optional, Iterator, List
import aiohttp
import numpy as np

from datasetstream.utils import get_np_dtype, unpack_batches


class DatasetClientIteratorAsync:
    """
    A single iterator instance for streaming data from a dataset,
    with asynchronous background prefetching.
    Each yielded item is a list of NumPy arrays (one batch); for legacy single-tensor streams, the tensor is wrapped in a one-element list.
    """

    def __init__(
            self,
            stream_url: str,
            seed: int,
            batch_size: int,
            seq_len: int,
            prefetch_size: int = 32,

            shuffle: bool = True,
            seek_document_start: bool = True,
            stop_at_document_end: bool = True
    ):
        self.stream_url = stream_url
        self.seed = seed
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.prefetch_size = prefetch_size

        self.shuffle = shuffle
        self.seek_document_start = seek_document_start
        self.stop_at_document_end = stop_at_document_end

        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None

        # Delay creation of the async buffer until __aenter__
        self._buffer: Optional[asyncio.Queue[List[np.ndarray]]] = None
        self._prefetch_task: Optional[asyncio.Task] = None

    async def __aenter__(self) -> 'DatasetClientIteratorAsync':
        # Create the buffer after the current loop is set (in __enter__)
        self._buffer = asyncio.Queue(maxsize=self.prefetch_size)

        self.session = aiohttp.ClientSession()
        try:
            self.ws = await self.session.ws_connect(
                self.stream_url,
                method='GET',
                headers={
                    'X-Iterator-Seed': str(self.seed),
                    'X-Iterator-BatchSize': str(self.batch_size),
                    'X-Iterator-SeqLen': str(self.seq_len),
                    'X-Iterator-Shuffle': "true" if self.shuffle else "false",
                    'X-Iterator-SeekDocumentStart': "true" if self.seek_document_start else "false",
                    'X-Iterator-StopAtDocumentEnd': "true" if self.stop_at_document_end else "false"
                },
                max_msg_size=1024 * 1024 * 1024
            )
            meta_data = await self.ws.receive_json()
            token_size_bytes = meta_data.get('token_size_bytes', None)
            if not token_size_bytes:
                raise ValueError("Missing token_size_bytes in dataset metadata")
            self.data_type = get_np_dtype(token_size_bytes)

            # Kick off the background prefetching task
            self._prefetch_task = asyncio.create_task(self._prefetch_loop())
            return self

        except Exception as e:
            if self.session:
                await self.session.close()
                self.session = None
            raise ConnectionError(f"Failed to connect to dataset stream: {str(e)}")

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Cancel the background task
        if self._prefetch_task:
            self._prefetch_task.cancel()
            try:
                await self._prefetch_task
            except asyncio.CancelledError:
                pass
            self._prefetch_task = None

        if self.ws:
            await self.ws.close()
            self.ws = None

        if self.session:
            await self.session.close()
            self.session = None

    async def _prefetch_loop(self):
        """
        Background task that continuously prefetches data and fills the buffer.
        """
        try:
            while True:
                # Determine how many items we can prefetch based on the queue capacity.
                # Since _buffer is now Optional (populated in __aenter__), we use assert.
                assert self._buffer is not None
                n_to_prefetch = self.prefetch_size - self._buffer.qsize()
                if n_to_prefetch <= 0:
                    # Sleep briefly to avoid busy waiting.
                    await asyncio.sleep(0.01)
                    continue

                # Request new items from the websocket
                await self.ws.send_json({'n_prefetch': n_to_prefetch})
                message = await self.ws.receive()

                if message.type == aiohttp.WSMsgType.BINARY:
                    batches = unpack_batches(message.data)
                    for batch in batches:
                        await self._buffer.put(batch)

                elif message.type in {
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                    aiohttp.WSMsgType.CLOSE,
                }:
                    # If the websocket is closed or encounters an error, exit the loop.
                    break
                else:
                    raise ValueError(f"Unexpected message type: {message.type}")
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully.
            pass
        except Exception as e:
            print(f"Error in prefetch loop: {str(e)}")

    def __aiter__(self) -> AsyncIterator[List[np.ndarray]]:
        return self

    async def __anext__(self) -> List[np.ndarray]:
        # If _buffer hasn't been created, something's wrong.
        if self._buffer is None:
            raise RuntimeError("Iterator not properly initialized.")
        next_item = await self._buffer.get()
        return next_item


class DatasetClientIteratorSync:
    """
    Synchronous wrapper for DatasetClientIteratorAsync.
    """

    def __init__(self, stream_url: str, seed: int, batch_size: int, seq_len: int, prefetch_size: int = 32,
                 shuffle: bool = True,
                 seek_document_start: bool = True,
                 stop_at_document_end: bool = True):
        self._async_iterator = DatasetClientIteratorAsync(
            stream_url, seed, batch_size, seq_len, prefetch_size,
            shuffle, seek_document_start, stop_at_document_end
        )
        self._loop = asyncio.new_event_loop()
        self._entered = False

    def __enter__(self) -> "DatasetClientIteratorSync":
        # Set the custom event loop as the current one.
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_iterator.__aenter__())
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._entered:
            self._loop.run_until_complete(self._async_iterator.__aexit__(exc_type, exc_val, exc_tb))
            self._entered = False
        self._loop.close()

    def __iter__(self) -> Iterator[List[np.ndarray]]:
        return self

    def __next__(self) -> List[np.ndarray]:
        try:
            return self._loop.run_until_complete(self._async_iterator.__anext__())
        except StopAsyncIteration:
            if self._entered:
                self._loop.run_until_complete(self._async_iterator.__aexit__(None, None, None))
                self._entered = False
            self._loop.close()
            raise StopIteration
