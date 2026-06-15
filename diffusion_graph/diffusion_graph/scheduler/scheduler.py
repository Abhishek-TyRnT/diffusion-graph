import torch
import torch.multiprocessing as mp
from typing import Any, Callable
from queue import Queue

from diffusion_graph.pipeline.pipeline_runner import DiffusionGraphRunner



# ── Worker functions (must be module-level for pickling) ────────────────────

def _worker_loop(name: str, method_name: str, lock: mp.Lock, q: mp.Queue):
    pipeline = q.get()
    method   = getattr(pipeline, method_name)

    while True:
        item = q.get()
        if item is None:
            break

        input_tensor, output_tensor, done_event = item
        #              ▲                ▲
        #              │                └─ mp.Event signals completion
        #              └─ both already in shared memory; no copy needed

        with lock:
            result = method(input_tensor)
            output_tensor.copy_(result)     # write directly into shared buffer

        done_event.set()  


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

class PipelineScheduler:
    """
    submit_pipeline()  — enqueue an input; returns immediately
    receive_pipeline() — generator; yields outputs in submission order
    """

    _OUTPUT_SHAPE = (1, 3, 512, 512)

    def __init__(self, pipeline: DiffusionPipeline):
        self._pipeline    = pipeline
        self._input_queue  = mp.Queue()
        self._output_queue = mp.Queue()
        self._next_seq_id  = 0

        # Single shared output buffer
        self._output_buffer = torch.empty(self._OUTPUT_SHAPE).share_memory_()

        # One lock + queue for the single generate worker
        self._lock = mp.Lock()
        self._worker_queue = mp.Queue()

        self._worker = mp.Process(
            target=PipelineScheduler._worker_loop,
            args=(self._lock, self._worker_queue),
            name="worker-generate",
            daemon=True,
        )
        self._worker.start()
        self._worker_queue.put(self._pipeline)      # bootstrap

        self._dispatcher = mp.Process(
            target=PipelineScheduler._dispatcher_loop,
            args=(
                self._lock,
                self._worker_queue,
                self._output_buffer,
                self._input_queue,
                self._output_queue,
            ),
            name="dispatcher",
            daemon=True,
        )
        self._dispatcher.start()

    # ── public API ───────────────────────────────────────────────────────────

    def submit_pipeline(self, image: torch.Tensor) -> int:
        """Enqueue an input. Returns immediately with the sequence id."""
        image.share_memory_()
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        self._input_queue.put((seq_id, image))
        return seq_id

    def receive_pipeline(self, total: int) -> Generator[torch.Tensor, None, None]:
        """
        Generator. Yields exactly `total` outputs in submission order.
        Blocks until each result is ready.
        """
        next_expected = 0
        pending: dict[int, torch.Tensor] = {}

        while next_expected < total:
            seq_id, result = self._output_queue.get()
            pending[seq_id] = result                # already cloned by dispatcher

            while next_expected in pending:
                yield pending.pop(next_expected)
                next_expected += 1

    def shutdown(self):
        self._input_queue.put(None)     # stop dispatcher
        self._dispatcher.join()
        self._worker_queue.put(None)    # stop worker
        self._worker.join()

    # ── dispatcher ───────────────────────────────────────────────────────────

    @staticmethod
    def _dispatcher_loop(
        lock:          mp.Lock,
        worker_queue:  mp.Queue,
        output_buffer: torch.Tensor,
        input_queue:   mp.Queue,
        output_queue:  mp.Queue,
    ):
        while True:
            item = input_queue.get()        # blocks here when idle
            if item is None:
                break

            seq_id, image = item

            done = mp.Event()
            worker_queue.put((image, output_buffer, done))

            with lock:
                done.wait()

            output_queue.put((seq_id, output_buffer.clone()))

    # ── worker ───────────────────────────────────────────────────────────────

    @staticmethod
    def _worker_loop(lock: mp.Lock, q: mp.Queue):
        pipeline = q.get()

        while True:
            item = q.get()
            if item is None:
                break

            input_tensor, output_tensor, done_event = item

            with lock:
                result = pipeline.generate(input_tensor)
                output_tensor.copy_(result)

            done_event.set()