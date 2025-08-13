import asyncio
import sys
from pathlib import Path
from typing import List

import numpy as np
from aiohttp import web
from aiohttp import WSMsgType

from datasetstream.config import ServerConfig, ServerState, ConfigError
from datasetstream.dataset import TokenDataset, CompoundDataset, CompoundDatasetIterator
from datasetstream.utils import get_np_dtype, pack_batches, unpack_batches


class DatasetServer:
    """Server for dataset streaming and monitoring"""

    def __init__(self, config_path: Path):
        """Initialize the dataset server

        :param config_path: Path to server configuration file
        :raises ConfigError: If configuration is invalid
        """
        self.config = ServerConfig.from_json(config_path)

        self.datasets = {}
        for dataset_id, dataset_config in self.config.dataset_configs.items():
            self.datasets[dataset_id] = CompoundDataset(
                [TokenDataset(file_path, dataset_config.token_size_bits, dataset_config.tokenizer_config.document_separator_token) for file_path in dataset_config.data_files]
            )

        self.state = ServerState(self.config)
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self) -> None:
        """Configure server routes"""
        # HTTP endpoints for monitoring
        self.app.router.add_get("/api/v1/status", self.handle_status)
        self.app.router.add_get("/api/v1/datasets", self.handle_list_datasets)

        # WebSocket endpoint for streaming data
        self.app.router.add_get("/api/v1/datasets/{dataset_id}/stream", self.handle_stream)

    async def handle_status(self, request: web.Request) -> web.Response:
        """Handle status endpoint - show server status

        :param request: HTTP request
        :return: JSON response with server status
        """
        return web.json_response({
            "status": "running",
            "host": self.config.host,
            "port": self.config.port,
            "active_connections": self.state.active_connections,
            "num_datasets": len(self.state.list_datasets())
        })

    async def handle_list_datasets(self, request: web.Request) -> web.Response:
        """Handle datasets endpoint - list available datasets

        :param request: HTTP request
        :return: JSON response with dataset info
        """
        return web.json_response({
            "datasets": self.state.list_datasets()
        })

    async def handle_stream(self, request: web.Request):
        """Handle WebSocket stream endpoint
        :param request: HTTP request
        :return: WebSocket response
        """
        # Get dataset id from URL path
        dataset_id = request.match_info['dataset_id']

        if dataset_id not in self.state.list_datasets():
            return web.json_response({
                "error": f"Dataset '{dataset_id}' not found"
            }, status=404)

        seed_header = request.headers.get("X-Iterator-Seed")
        batch_size_header = request.headers.get("X-Iterator-BatchSize")
        seq_len_header = request.headers.get("X-Iterator-SeqLen")

        if not seed_header or not batch_size_header or not seq_len_header:
            return web.json_response({
                "error": "Missing required headers: X-Iterator-Seed, X-Iterator-BatchSize, X-Iterator-SeqLen"
            }, status=400)

        seed = int(seed_header)
        batch_size = int(batch_size_header)
        seq_len = int(seq_len_header)

        dataset = self.datasets[dataset_id]
        dataset_iterator = CompoundDatasetIterator(dataset, batch_size, seq_len, seed, shuffle=True)

        # Add connection to active connections
        self.state.increment_connections(dataset_id)

        ws = web.WebSocketResponse(max_msg_size=1024 * 1024 * 1024) # insane max size; RFC says this is fine. Sue me
        await ws.prepare(request)

        print(f"New connection to dataset {dataset_id} with seed={seed}, batch_size={batch_size}")

        # determine token size in bytes
        dataset_config = self.config.dataset_configs[dataset_id]
        n_bits = dataset_config.token_size_bits

        token_size_bytes = 0
        if n_bits <= 8:
            token_size_bytes = 1
        elif n_bits <= 16:
            token_size_bytes = 2
        elif n_bits <= 32:
            token_size_bytes = 4
        elif n_bits <= 64:
            token_size_bytes = 8

        await ws.send_json({'token_size_bytes': token_size_bytes})

        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                if msg.data == "close":
                    await ws.close()
                else:
                    req = msg.json()
                    n_prefetch = req.get("n_prefetch", 1)

                    batches: List[List[np.ndarray]] = []
                    for _ in range(n_prefetch):
                        next_batch = list(next(dataset_iterator))  # list of ndarrays
                        batches.append(next_batch)

                    payload = pack_batches(batches)
                    await ws.send_bytes(payload)

            elif msg.type == WSMsgType.ERROR:
                print(f"ws connection closed with exception {ws.exception()}")

        # Remove connection from active connections
        self.state.decrement_connections(dataset_id)
        return ws

    async def run(self) -> None:
        """Run the dataset server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port, reuse_port=True)

        await site.start()
        print(f"Dataset server running at http://{self.config.host}:{self.config.port}")
        print(f"Available datasets: {list(self.state.list_datasets().keys())}")

        # Run forever
        try:
            await asyncio.Future()
        finally:
            await runner.cleanup()


def main() -> None:
    """Main entry point for the dataset server"""
    if len(sys.argv) != 2:
        print("Usage: dataset_server <config_path>")
        sys.exit(1)

    try:
        config_path = Path(sys.argv[1])
        server = DatasetServer(config_path)
        asyncio.run(server.run())

    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
