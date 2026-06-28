import torch
import uuid
import torch.multiprocessing as mp
from enum import Enum
import time

from typing import Any, Callable

from diffusion_graph.pipeline.pipeline_runner import DiffusionGraphRunner



class Status(Enum):
    TODO = 1
    IN_PROGRESS = 2
    DONE = 3    
#TODO: Use the TensorQueue for managing data transfer between processes
# class TensorQueue:
#     def __init__(self,):
#         self.prompt_queue = Queue(maxsize=5)
#         self.negative_prompt_queue = Queue(maxsize=5)
        
#     def copy_to_buffer(self, buffer: torch.Tensor, queue_name: str):

#         if hasattr(self, queue_name):
#             queue = getattr(self, queue_name)
#             tensor = queue.get()
#             buffer.copy_(tensor)
#             return 
        
#         raise ValueError(f"Queue {queue_name} does not exist")

#     def submit_queue(self, tensor: torch.Tensor, queue_name: str):

#         if hasattr(self, queue_name):
#             queue = getattr(self, queue_name)
#             queue.put(tensor)
#             return 
        
#         raise ValueError(f"Queue {queue_name} does not exist")
        
        

# ── Scheduler ────────────────────────────────────────────────────────────────

class DiffusionGraphScheduler:
    """
    submit_pipeline()  — enqueue an input; returns immediately
    receive_pipeline() — generator; yields outputs in submission order
    """

    def __init__(self, artifact_directory: str, device:str, tokenizer:str):
        mp.set_start_method("spawn")
        self._pipeline = DiffusionGraphRunner(artifact_directory, device, tokenizer)

        self._input_queue  = mp.Queue()
        self._output_queue = mp.Queue()

        self._result_queue = mp.Queue()
        self._next_seq_id  = 0

        self.global_state: dict = {}
        # One lock + queue for the single generate worker
        self._worker_queue = mp.Queue()

        print("Starting worker process...")
        self._worker = mp.Process(
            target=DiffusionGraphScheduler._worker_loop,
            args=(self._worker_queue, self._result_queue),
            name="worker-generate",
            daemon=True,
        )
        self._worker.start()
        self._worker_queue.put(self._pipeline)      # bootstrap

        self._dispatcher = mp.Process(
            target=DiffusionGraphScheduler._dispatcher_loop,
            args=(
                self._worker_queue,
                self._result_queue,
                self._input_queue,
                self._output_queue,
            ),
            name="dispatcher",
            daemon=True,
        )

        print("Starting dispatcher process...")
        self._dispatcher.start()

    # ── public API ───────────────────────────────────────────────────────────

    def submit_pipeline(self, input_args: tuple) -> int:
        """Enqueue an input. Returns immediately with the sequence id."""
        seq_id = self._next_seq_id
        request_id = uuid.uuid4()
        self._next_seq_id += 1
        self._input_queue.put((seq_id, input_args, request_id))
        self.global_state[request_id] = Status.TODO
        print(f"Input submitted to the scheduler with seq_id: {seq_id}")
        return request_id

    def receive_pipeline(self, total: int):
        """
        Generator. Yields exactly `total` outputs in submission order.
        Blocks until each result is ready.
        """
        next_expected = 0
        pending: dict[int, torch.Tensor] = {}

        while next_expected < total:
            seq_id, result, request_id = self._output_queue.get()
            pending[seq_id] = result                # already cloned by dispatcher

            while next_expected in pending:
                yield pending.pop(next_expected)
                next_expected += 1

    def receive_by_request_id(self, request_id: uuid.UUID):
        if request_id not in self.global_state:
            raise ValueError(f"Request ID {request_id} not found.")
        seconds = 10

        iterations = 10
        i = 0
        while i < iterations and self.global_state[request_id] != Status.DONE:
            time.sleep(seconds)
            i += 1

        if i == iterations:
            return None
        
        total = self._output_queue.qsize()

        for _ in range(total):
            seq_id, result, curr_request_id = self._output_queue.get()
            if curr_request_id == request_id:
                return result
        
        return None
        
    def shutdown(self):
        self._input_queue.put(None)     # stop dispatcher
        self._dispatcher.join()
        self._worker_queue.put(None)    # stop worker
        self._worker.join()

    # ── dispatcher ───────────────────────────────────────────────────────────

    @staticmethod
    def _dispatcher_loop(
        worker_queue:  mp.Queue,
        result_queue:  mp.Queue,
        input_queue:   mp.Queue,
        output_queue:  mp.Queue,
    ):
        while True:
            item = input_queue.get()
            if item is None:
                break

            seq_id, input_args, request_id = item
            worker_queue.put(input_args)
            self.global_state[request_id] = Status.IN_PROGRESS
            result = result_queue.get()          # blocks until worker is done
            output_queue.put((seq_id, result, request_id))
            self.global_state[request_id] = Status.DONE

    # ── worker ───────────────────────────────────────────────────────────────

    @staticmethod
    def _worker_loop(worker_queue: mp.Queue, result_queue: mp.Queue):
        pipeline = worker_queue.get()
        pipeline.load_pipeline()
        
        while True:
            item = worker_queue.get()
            if item is None:
                break

            extra_kwargs = item[-1]
            input_args = item[:-1]

            result = pipeline.generate(*input_args, **extra_kwargs)
            result_queue.put(result)