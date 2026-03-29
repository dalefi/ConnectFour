import asyncio
import time

import torch


class NeuralNetBatcher:
    def __init__(self, model, device, batch_size=32, timeout=0.02):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = []
        self._lock = asyncio.Lock()
        self._event_loop = asyncio.get_event_loop()
        self.batch_stats = {"count": 0, "total_size": 0, "timeouts": 0}
        self.start_time = time.time()
        self.total_predicted = 0

    async def get_policy_value(self, state_tensor):
        future = self._event_loop.create_future()
        should_process = False

        async with self._lock:
            self.queue.append((state_tensor, future))
            if len(self.queue) >= self.batch_size:
                should_process = True
            elif len(self.queue) == 1:
                self._event_loop.call_later(self.timeout, self._process_batch_sync)

        if should_process:
            await self._process_batch()

        return await future

    def _process_batch_sync(self):
        asyncio.create_task(self._process_batch())

    async def _process_batch(self):
        async with self._lock:
            if not self.queue:
                return
            current_batch = self.queue
            self.queue = []

        states, futures = zip(*current_batch)
        batch_tensor = torch.stack(states).to(self.device)

        with torch.no_grad():
            out = self.model(batch_tensor)
            policies = torch.nn.functional.softmax(out['policy'], dim=1)
            values = out['value']

        for i, future in enumerate(futures):
            if not future.done():
                future.set_result((values[i].cpu().item(), policies[i].cpu().numpy()))

        # size = len(current_batch)
        # print(f"[BATCHER] Batch of {size} completed.")
        # self.total_predicted += size
        # if self.batch_stats["count"] % 50 == 0:
        #     elapsed = time.time() - self.start_time
        #     # Predictions pro Sekunde sind ein guter Proxy für MPS
        #     pps = self.total_predicted / elapsed
        #     print(f" >>> GPU Throughput: {pps:.1f} predictions/s")

    def print_stats(self):
        if self.batch_stats["count"] == 0: return
        avg_size = self.batch_stats["total_size"] / self.batch_stats["count"]
        timeout_rate = (self.batch_stats["timeouts"] / self.batch_stats["count"]) * 100
        print(f"[Batcher] Avg Size: {avg_size:.1f}/{self.batch_size} | Timeouts: {timeout_rate:.1f}%")