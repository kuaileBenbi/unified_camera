import threading
import numpy as np
from rknnlite.api import RKNNLite
from concurrent.futures import Future, ThreadPoolExecutor


def initRKNNs(det_model: str, num: int):
    pool = []
    core_map = [
        RKNNLite.NPU_CORE_0,
        RKNNLite.NPU_CORE_1,
        RKNNLite.NPU_CORE_2,
        RKNNLite.NPU_CORE_0_1_2,
    ]

    for i in range(num):
        core_mask = core_map[i % len(core_map)]
        rknn = RKNNLite()
        if rknn.load_rknn(det_model) != 0:
            raise RuntimeError(f"加载 RKNN 模型失败: {det_model}")
        if rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError(f"初始化 RKNN 运行时失败（core {core_mask}）")
        pool.append(rknn)

    return pool


class detectExecutor:
    """
    基于 ThreadPoolExecutor + Semaphore 限流的检测执行器，
    避免提交端过快导致任务无限积压。
    """

    def __init__(
        self,
        det_model: str,
        TPEs: int,
        func,
        callback,
        max_pending_tasks: int = None,
        drop_when_full: bool = True,
    ):
        """
        Args:
            det_model: 模型路径或对象
            TPEs: 并行实例/线程数
            func: 检测函数签名 func(rknn_instance, frame) -> result
            callback: 结果回调签名 callback(result, frame_id)
            max_pending_tasks: 信号量容量，表示同时允许多少任务在队列中等待
                               (包括正在跑和已经提交未调度的)
                               默认 = TPEs * 2
            drop_when_full: 信号量被占满时，True 则直接丢帧，False 则阻塞直到有空位
        """

        self.TPEs = TPEs
        self.func = func
        self.callback = callback

        # 初始化 RKNN 实例池和线程池
        self.rknnPool = initRKNNs(det_model, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self._round_robin = 0

        # 用来记录每帧提交时的时间戳（ns）
        # self.start_times: Dict[int, int] = {}

        # 信号量限流，避免无限堆积
        if max_pending_tasks is None:
            max_pending_tasks = TPEs * 2
        self.sema = threading.Semaphore(max_pending_tasks)
        self.drop_when_full = drop_when_full

    def put(self, frame: np.ndarray, frame_id: int = None):
        """
        提交一个检测任务。
        若当前待处理任务数 >= max_pending_tasks:
          - drop_when_full=True 时直接返回丢帧
          - drop_when_full=False 时阻塞直到有空位
        """
        # 尝试获取信号量
        acquired = self.sema.acquire(blocking=not self.drop_when_full)
        if not acquired:
            # 丢帧策略：信号量满且不阻塞，直接丢掉这一帧
            return

        # 记录提交时刻
        # self.start_times[frame_id] = time.perf_counter_ns()

        # 轮询选一个 RKNN 实例
        idx = self._round_robin
        rknn_ins = self.rknnPool[idx]
        self._round_robin = (idx + 1) % self.TPEs

        # 提交到线程池
        fut: Future = self.pool.submit(self._worker, rknn_ins, frame, frame_id)
        # 用 wrap_on_done 保证最终都会释放信号量
        fut.add_done_callback(lambda f: self._wrap_on_done(f))

    def _worker(self, rknn_ins: RKNNLite, frame: np.ndarray, frame_id: int):
        """
        真正跑推理的函数，返回：
            (frame_id, results, t_worker_start, t_worker_end)
        """
        # t_worker_start = time.perf_counter_ns()
        results = self.func(rknn_ins, frame)
        # t_worker_end = time.perf_counter_ns()
        # return frame_id, results, t_worker_start, t_worker_end
        return frame_id, results

    def _wrap_on_done(self, fut: Future):
        """
        包装后的回调：先释放信号量，再调 _on_done
        """
        try:
            self._on_done(fut)
        finally:
            # 一定要在回调完成后释放信号量
            self.sema.release()

    def _on_done(self, fut: Future):
        """
        内部回调，获取 future 结果、计算各种时序，并调用用户回调。
        """
        try:
            # frame_id, results, t_ws, t_we = fut.result()
            frame_id, results = fut.result()
        except Exception as e:
            import traceback

            print(f"[detectExecutor] 任务执行异常: {traceback.print_exc()}")
            return

        # t_main_cb = time.perf_counter_ns()

        # 1) 端到端耗时（从 put 到 callback）
        # t0 = self.start_times.pop(frame_id, None)
        # end_to_end_ms = (t_main_cb - t0) / 1e6 if t0 else None

        # 2) 真正推理耗时
        # processing_ms = (t_we - t_ws) / 1e6

        # 3) 调度+通信耗时（推理结束到回调开始）
        # comm_sched_ms = (t_main_cb - t_we) / 1e6

        # print(
        #     f"[Frame {frame_id}] E2E={end_to_end_ms:.1f}ms",
        #     f"proc={processing_ms:.1f}ms",
        #     f"comm+sched={comm_sched_ms:.1f}ms",
        # )

        # 调用用户回调
        try:
            self.callback(results, frame_id)
        except Exception:
            import traceback

            print(f"[detectExecutor] 用户回调异常:\n{traceback.print_exc()}")

    def release(self):
        """
        关停线程池和释放 RKNN 资源
        """
        self.pool.shutdown(wait=True)
        for inst in self.rknnPool:
            inst.release()
