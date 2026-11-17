"""调度中枢与连续批处理 (Continuous Batching) 单元测试 (TDD 规格说明书)。

本测试文件作为 Scheduler 的可执行契约规范，涵盖:
1. 调度器初始化、空闲槽位池 (Slot Pool) 与基础属性契约;
2. 单请求全生命周期 (WAITING -> RUNNING -> FINISHED) 与物理槽位绑定/回收;
3. 并发容量与槽位配额限制 (max_batch_size 满载排队与背压);
4. 连续批处理核心机制: 短请求命中 EOS 释放槽位，等待请求即刻动态插队复用 (Slot Reuse);
5. 主动取消 (Abort) 请求时的槽位零泄漏回收;
6. 边界与空队列调度防护。
"""

import pytest
import torch

from stepinfer.scheduler.request import Request, RequestStatus
from stepinfer.scheduler.request_queue import QueueFullError, RequestQueue
from stepinfer.scheduler.scheduler import ScheduleResult, Scheduler


# ==============================================================================
# 场景 1: 初始化与空闲槽位池契约
# ==============================================================================


def test_scheduler_initialization_and_slot_pool():
    """【场景 1】验证调度器初始化与物理插槽池 (Slot Pool) 契约。

    核心考量:
        连续批处理引擎的核心是预先分配的一组物理计算插槽 (0 到 max_batch_size - 1)。
        调度器初始化时，所有插槽均应处于空闲 (free) 状态。
    """
    max_batch_size = 4
    max_seq_len = 1024
    scheduler = Scheduler(max_batch_size=max_batch_size, max_seq_len=max_seq_len)

    # 1. 基础配置
    assert scheduler.max_batch_size == max_batch_size
    assert scheduler.max_seq_len == max_seq_len

    # 2. 初始状态：无运行请求，无排队请求
    assert scheduler.num_running_requests == 0
    assert scheduler.num_waiting_requests == 0
    assert not scheduler.has_unfinished_requests()

    # 3. 槽位池契约：初始时可用槽位数必须等于 max_batch_size
    # 期望维护 free_slots 集合，包含 {0, 1, 2, 3}
    assert hasattr(scheduler, "free_slots"), "Scheduler 必须维护物理空闲插槽池 free_slots"
    assert scheduler.free_slots == {0, 1, 2, 3}


def test_scheduler_rejects_invalid_init_parameters():
    """验证初始化参数合法性防御。"""
    with pytest.raises(ValueError, match="max_batch_size must be a positive integer"):
        Scheduler(max_batch_size=0)

    with pytest.raises(ValueError, match="max_seq_len must be a positive integer"):
        Scheduler(max_batch_size=4, max_seq_len=-10)


# ==============================================================================
# 场景 2: 请求准入与超长拦截
# ==============================================================================


def test_add_request_and_seq_len_guard():
    """【场景 2】验证提交请求时的队列排队与上下文长度超界防御。"""
    scheduler = Scheduler(max_batch_size=2, max_seq_len=10)

    # 1. 提交合规请求
    req_valid = Request(request_id="req-1", prompt="hi", prompt_token_ids=[1, 2, 3])
    scheduler.add_request(req_valid)
    assert scheduler.num_waiting_requests == 1
    assert req_valid.status == RequestStatus.WAITING

    # 2. 提交超出 max_seq_len 的超长请求 -> 必须在入口当场拦截抛出 ValueError
    req_too_long = Request(
        request_id="req-bad",
        prompt="too long",
        prompt_token_ids=list(range(20)),  # 长度 20 > max_seq_len 10
    )
    with pytest.raises(ValueError, match="exceeds model max_seq_len"):
        scheduler.add_request(req_too_long)


# ==============================================================================
# 场景 3: 单请求生命周期与物理插槽全流程绑定与回收
# ==============================================================================


def test_single_request_lifecycle_and_slot_management():
    """【场景 3】端到端跟踪单个请求的生命周期与插槽变化。

    时间线演进:
        Step 0: 请求提交入队 (WAITING, slot_id=None)
        Step 1: schedule() 调度 Prefill
                -> 请求状态流转为 RUNNING
                -> 分配一个物理 slot_id (如 0)
                -> free_slots 减少该 slot_id
        Step 2: post_step() 写回首个 Token
                -> 此时请求尚未完成，slot_id 必须继续保持占用
        Step 3: schedule() 调度 Decode 步
        Step 4: post_step() 写回最终 Token (达 max_new_tokens 触发结束)
                -> 请求状态流转为 FINISHED
                -> 移出 running_requests
                -> slot_id 释放归还给 free_slots！
    """
    scheduler = Scheduler(max_batch_size=2, max_seq_len=128)
    req = Request(
        request_id="req-1",
        prompt="Hello",
        prompt_token_ids=[10, 20],
        max_new_tokens=2,  # 仅生成 2 个 Token 即可完成
    )
    scheduler.add_request(req)

    # --------------------------------------------------------------------------
    # Step 1: 首步调度 (Prefill)
    # --------------------------------------------------------------------------
    result = scheduler.schedule()

    assert not result.is_empty
    assert result.is_prefill is True
    assert result.batch_size == 1
    assert result.requests[0].request_id == "req-1"

    # 关键断言: 请求进入 RUNNING，并被成功分配了物理 slot_id
    assert req.status == RequestStatus.RUNNING
    assert req.slot_id is not None
    allocated_slot = req.slot_id
    assert 0 <= allocated_slot < 2

    # 空闲槽位池应当扣除该插槽
    assert allocated_slot not in scheduler.free_slots
    assert len(scheduler.free_slots) == 1
    assert scheduler.num_running_requests == 1

    # --------------------------------------------------------------------------
    # Step 2: 模型完成首步 Prefill 前向，通过 post_step 注入 Token 101
    # --------------------------------------------------------------------------
    finished = scheduler.post_step(requests=[req], next_tokens=[101])
    assert finished == []  # 生成 1 个，上限 2 个，未完成
    assert req.generated_token_ids == [101]
    assert not req.is_finished
    # 插槽依然被持有
    assert req.slot_id == allocated_slot
    assert allocated_slot not in scheduler.free_slots

    # --------------------------------------------------------------------------
    # Step 3: 次步调度 (Decode)
    # --------------------------------------------------------------------------
    result_decode = scheduler.schedule()
    assert not result_decode.is_empty
    assert result_decode.is_prefill is False
    assert result_decode.batch_size == 1

    # --------------------------------------------------------------------------
    # Step 4: 模型完成单步 Decode 前向，注入 Token 102 (达到上限 2)
    # --------------------------------------------------------------------------
    finished = scheduler.post_step(requests=[req], next_tokens=[102])

    assert len(finished) == 1
    assert finished[0].request_id == "req-1"
    assert req.is_finished
    assert req.stop_reason == "length"

    # 关键断言: 已完成请求从 running_requests 移出，其持有的物理 slot_id 必须物归原主！
    assert scheduler.num_running_requests == 0
    assert allocated_slot in scheduler.free_slots
    assert scheduler.free_slots == {0, 1}
    assert not scheduler.has_unfinished_requests()


# ==============================================================================
# 场景 4: 槽位配额满载与排队拦截
# ==============================================================================


def test_scheduler_respects_max_batch_size_and_queuing():
    """【场景 4】验证并发槽位耗尽时，后续请求停留在队列等待。

    设定 max_batch_size = 2，但连续灌入 3 个请求：
        - Req 1, Req 2 应该被调度并分别分配不同的 Slot；
        - Req 3 必须留在等待队列中，绝对不能强行抢占或越界。
    """
    scheduler = Scheduler(max_batch_size=2, max_seq_len=64)

    req1 = Request("req-1", "p1", [1, 2], max_new_tokens=5)
    req2 = Request("req-2", "p2", [3, 4], max_new_tokens=5)
    req3 = Request("req-3", "p3", [5, 6], max_new_tokens=5)

    scheduler.add_request(req1)
    scheduler.add_request(req2)
    scheduler.add_request(req3)

    assert scheduler.num_waiting_requests == 3

    # 执行首次调度
    result = scheduler.schedule()

    # 只能容纳 2 个请求
    assert result.batch_size == 2
    scheduled_ids = {r.request_id for r in result.requests}
    assert scheduled_ids == {"req-1", "req-2"}

    # 两个请求占有不同插槽
    assert req1.slot_id is not None
    assert req2.slot_id is not None
    assert req1.slot_id != req2.slot_id
    assert {req1.slot_id, req2.slot_id} == {0, 1}

    # 空闲池已经耗尽
    assert len(scheduler.free_slots) == 0

    # Req 3 必须在等待队列中继续等待，且未被分配 slot
    assert scheduler.num_waiting_requests == 1
    assert req3.status == RequestStatus.WAITING
    assert req3.slot_id is None


# ==============================================================================
# 场景 5: 连续批处理的核心灵魂——短请求退场，等待请求即刻填补插槽 (Slot Reuse)
# ==============================================================================


def test_continuous_batching_dynamic_slot_reuse():
    """【场景 5】连续批处理核心: 序列动态退场与即刻插槽复用。

    物理过程图解:
        Slot 0: [Req A (短，跑 1 步遇 EOS)] ──► 触发 EOS 退场 ──► 释放 Slot 0 ──► [Req C 抢占 Slot 0 入场]
        Slot 1: [Req B (长，跑 5 步)] ───────────────────────────────► 持续解码 (不受任何干扰)

    本测试验证:
        1. 命中 EOS 的序列当场终止并释放插槽；
        2. 空出的插槽在下一轮调度时被排队请求平滑继承；
        3. 两序列在不同槽位交替前向，游标与数据互不踩踏。
    """
    scheduler = Scheduler(max_batch_size=2, max_seq_len=64)
    eos_id = 999

    req_a = Request("req-A", "short", [10], max_new_tokens=10)
    req_b = Request("req-B", "long", [20], max_new_tokens=10)
    req_c = Request("req-C", "queued", [30], max_new_tokens=10)

    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    scheduler.add_request(req_c)

    # 1. 第一步 Prefill: Req A 和 Req B 成功瓜分 Slot 0 和 Slot 1
    res1 = scheduler.schedule()
    assert res1.batch_size == 2
    slot_a = req_a.slot_id
    slot_b = req_b.slot_id
    assert {slot_a, slot_b} == {0, 1}

    # 2. 模拟前向: Req A 吐出 EOS (999)，Req B 吐出普通 Token (50)
    finished = scheduler.post_step(
        requests=[req_a, req_b],
        next_tokens=[eos_id, 50],
        eos_token_id=eos_id,
    )

    # 验证: Req A 触发完成并退场，Req B 仍在进行
    assert len(finished) == 1
    assert finished[0].request_id == "req-A"
    assert req_a.is_finished
    assert req_a.stop_reason == "stop"
    assert not req_b.is_finished

    # 验证: Req A 的插槽已经被回收进 free_slots！
    assert slot_a in scheduler.free_slots
    assert scheduler.num_running_requests == 1
    assert "req-A" not in scheduler.running_requests

    # 3. 接下来最关键的调度步 (连续批处理核心):
    # 此时系统中:
    #   - Req B 处于 RUNNING (占据 slot_b)
    #   - Req C 处于 WAITING
    #   - free_slots 中刚好空出 slot_a
    # 当调度器再次调度时:
    # 策略 A: 运行池里的 Req B 进入单步 Decode
    # 并且如果有空闲插槽，排队的 Req C 应当被拉取入场进行 Prefill (或在此步/下一步完成组装)
    # 此时如果调度 Prefill 新请求:
    if scheduler.free_slots and not scheduler.request_queue.is_empty():
        # 如果是两阶段解耦调度: 优先满足 Decode 或允许新请求插队
        pass

    # 让我们主动推进调度，直到 Req C 获得插槽
    # 无论调度策略是先处理当前 Decode 还是立即调度新 Prefill:
    # 只要 Req C 获得运行资格，它分配到的 slot_id 必然是刚才 Req A 释放出来的 slot_a！
    while req_c.status == RequestStatus.WAITING:
        step_res = scheduler.schedule()
        if req_c in step_res.requests:
            break
        # 推进老请求解码
        scheduler.post_step(step_res.requests, next_tokens=[51] * len(step_res.requests))

    # 断言: Req C 成功夺取了之前属于 Req A 的槽位！
    assert req_c.status == RequestStatus.RUNNING
    assert req_c.slot_id == slot_a
    assert req_c.slot_id != req_b.slot_id


# ==============================================================================
# 场景 6: 客户端主动取消 (Abort) 时的插槽零泄漏
# ==============================================================================


def test_abort_running_request_frees_allocated_slot():
    """【场景 6】验证运行中请求被外部主动 Abort 取消时，物理插槽能无条件干净释放。"""
    scheduler = Scheduler(max_batch_size=2, max_seq_len=64)
    req = Request("req-cancel", "prompt", [1, 2])
    scheduler.add_request(req)

    # 调度入场
    scheduler.schedule()
    assert req.status == RequestStatus.RUNNING
    slot_id = req.slot_id
    assert slot_id is not None
    assert slot_id not in scheduler.free_slots

    # 客户端主动取消该请求
    aborted = scheduler.abort_request("req-cancel")
    assert aborted is True
    assert req.is_finished
    assert req.stop_reason == "aborted"

    # 核心契约: 该请求从 running 移除，且 slot_id 必须归还给 free_slots
    assert "req-cancel" not in scheduler.running_requests
    assert slot_id in scheduler.free_slots
    assert scheduler.free_slots == {0, 1}
    assert scheduler.num_running_requests == 0


# ==============================================================================
# 场景 7: 边界保护与空队列
# ==============================================================================


def test_scheduler_empty_schedule_returns_empty_result():
    """【场景 7】验证队列与运行池全空时，schedule() 安全返回 is_empty=True。"""
    scheduler = Scheduler(max_batch_size=4)
    result = scheduler.schedule()

    assert result.is_empty
    assert result.batch_size == 0
    assert result.requests == []
    assert result.num_tokens == 0
