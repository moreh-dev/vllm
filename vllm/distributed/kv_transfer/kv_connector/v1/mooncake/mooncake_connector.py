# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TpKVTopology,
    get_current_attn_backend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_local_first_rank,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    raise ImportError(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
        "to run VLLM with MooncakeTransferEngine."
    ) from e

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)

logger = init_logger(__name__)


def _producer_trace_enabled() -> bool:
    return os.environ.get("VLLM_MOONCAKE_PRODUCER_TRACE", "0") == "1"


def _producer_trace(msg: str, *args: object) -> None:
    if _producer_trace_enabled():
        logger.info("[ProducerTrace] " + msg, *args)


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    kv_caches_base_addr: list[int]
    # Layer 2' (READ pull): if non-empty, this is an ACK for transfers already
    # read-completed by the consumer. Producer uses this to free blocks.
    ack_for_transfers: list[TransferId] = []


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2
    # Layer 2' ACK was processed by producer.
    ACK_OK = 3


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None
    # Layer 2' (READ pull): when set, consumer must perform
    # batch_transfer_sync_read(p_session, p_dst_ptrs, p_src_ptrs, p_lengths).
    # The sync_read return guarantees GPU memory landing (cudaMemcpy synced).
    p_session: str | None = None
    p_src_ptrs: list[int] | None = None
    p_dst_ptrs: list[int] | None = None
    p_lengths: list[int] | None = None


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    # Set expire time to avoid infinitely sending requests.
    expire_time: float = float("inf")
    # Designed for one D pairing to multiple P
    pull_tasks_count: int = 0


@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0
    acked_remote_tp_ranks: set[int] | None = None


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # Use (engine_id, dp_rank) to group reqs with same dp.
        # See comments in MooncakeBootstrapServer.
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[int]]] = {}
        self.reqs_not_processed: set[TransferId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        transfer_id = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id = kv_transfer_params["remote_engine_id"]
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_id=transfer_id,
            )
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)


class MooncakeConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, self.engine_id)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        pass


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config

        assert vllm_config.kv_transfer_config
        self.is_kv_producer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, tuple[Request, list[int]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[TransferId] = set()

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if not params:
            return 0, False

        if params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert not self.is_kv_producer
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "req_id=%s num_external_tokens=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            if all(
                p in params
                for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")
            ):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                local_block_ids = (
                    blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
                )
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            assert not self.is_kv_consumer
            if not params.get("transfer_id"):
                logger.warning("Missing transfer_id in kv_transfer_params from router!")
            else:
                # Add an empty list to worker to create event.
                self._reqs_need_send[request.request_id] = (request, [])
                _producer_trace(
                    "prefill_enqueue req=%s transfer=%s prompt_tokens=%d "
                    "external_tokens=%d placeholder_blocks=0",
                    request.request_id,
                    params["transfer_id"],
                    len(request.prompt_token_ids or []),
                    num_external_tokens,
                )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to PullReqMeta.
        if not self.is_kv_producer:
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if not self.is_kv_consumer:
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=False,
                )
                _producer_trace(
                    "build_meta_send req=%s transfer=%s blocks=%d",
                    req_id,
                    req.kv_transfer_params["transfer_id"],
                    len(block_ids),
                )
            self._reqs_need_send.clear()
            meta.reqs_not_processed = self._reqs_not_processed
            self._reqs_not_processed = set()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, req_id=%s, request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params or not params.get("transfer_id"):
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert not self.is_kv_producer
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None

        assert not self.is_kv_consumer

        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(params["transfer_id"])
            _producer_trace(
                "prefill_drop req=%s transfer=%s status=%s",
                request.request_id,
                params["transfer_id"],
                request.status,
            )
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = (request, block_ids)
            _producer_trace(
                "prefill_ready req=%s transfer=%s blocks=%d",
                request.request_id,
                params["transfer_id"],
                len(block_ids),
            )

        return delay_free_blocks, None


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config

        self.engine = TransferEngine()
        self.hostname = get_ip()

        assert (kv_transfer_config := vllm_config.kv_transfer_config)
        self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
        self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )
        # Create more tasks than workers to keep the thread pool saturated.
        # Tasks can await async events, so a surplus (2x is a robust heuristic)
        # prevents workers from idling.
        self.num_sender_tasks = self.num_sender_workers * 2
        protocol = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
            "mooncake_protocol", "rdma"
        )
        logger.info(
            "The Mooncake Transfer Engine is using %s as its protocol.", protocol
        )
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
        self._pending_bootstrap_querys: dict[str, asyncio.Event] = {}
        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        try:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        except AssertionError:
            # TT backend runs one vLLM worker over a multi-device mesh and does
            # not initialize vLLM TP group. Treat it as one logical worker;
            # virtual TP registration below advertises GPU-facing TP ranks.
            logger.warning(
                "Mooncake worker using TP fallback rank=0 size=1; "
                "vLLM TP group is not initialized"
            )
            self.tp_rank = 0
            self.tp_size = 1
        self.num_blocks = 0

        assert (parallel_config := vllm_config.parallel_config)
        dp_rank = parallel_config.data_parallel_index
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            raise ValueError(
                "Mooncake Transfer Engine does not support pipeline parallelism yet."
            )
        try:
            self.pp_rank = get_pp_group().rank_in_group
        except AssertionError:
            self.pp_rank = 0

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}

        # For kv_both, we will act both prefiller and decoder.
        if not self.is_kv_consumer:
            # Background threads for sending kvcaches to D.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="vllm-mooncake-sender",
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches",
                self.num_sender_workers,
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

            # Start bootstrap server on global rank 0.
            if should_launch_bootstrap_server(vllm_config):
                _, port = get_mooncake_bootstrap_addr(vllm_config)
                # vLLM bootstrap server signature changed across versions.
                # Newer vLLM expects (host, port); older expects
                # (vllm_config, host, port).
                try:
                    self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", port)
                except TypeError:
                    self.bootstrap_server = MooncakeBootstrapServer(
                        vllm_config, "0.0.0.0", port
                    )
                self.bootstrap_server.start()

        if not self.is_kv_producer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        backend = get_current_attn_backend(vllm_config)
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
        self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}
        self.kv_topo = TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,  # shared state
            remote_block_size=self._block_size,  # shared state
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backend=backend,
        )

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.async_zmq_ctx.term()
        if not self.is_kv_consumer:
            self._sender_executor.shutdown(wait=False)
            if self.sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._sender_listener_t.join()
            if should_launch_bootstrap_server(self.vllm_config):
                self.bootstrap_server.shutdown()
        if not self.is_kv_producer and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    async def register_worker_with_bootstrap(self):
        host, port = get_mooncake_bootstrap_addr(self.vllm_config)
        url = make_zmq_path("http", host, port) + "/register"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        _producer_trace(
            "bootstrap_register_begin engine=%s worker=%s url=%s",
            self.engine_id,
            worker_addr,
            url,
        )
        # For NPU single-worker: register as multiple TP ranks so that
        # GPU consumers with higher TP can connect (heterogeneous TP workaround).
        # The actual KV data is already TP-gathered (full tensor).
        num_virtual_tp = int(os.environ.get("VLLM_MOONCAKE_VIRTUAL_TP_SIZE", "0"))
        tp_ranks_to_register = list(range(num_virtual_tp)) if num_virtual_tp > 0 else [self.tp_rank]
        for vtp_rank in tp_ranks_to_register:
            payload = RegisterWorkerPayload(
                engine_id=self.engine_id,
                dp_rank=self.dp_rank,
                tp_rank=vtp_rank,
                pp_rank=self.pp_rank,
                addr=worker_addr,
            )
            while True:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(url, json=payload.model_dump())
                        response.raise_for_status()
                    logger.debug("Registered tp_rank=%d with bootstrap server at %s", vtp_rank, url)
                    _producer_trace(
                        "bootstrap_register_ok tp_rank=%d worker=%s",
                        vtp_rank,
                        worker_addr,
                    )
                    break
                except httpx.ConnectError:
                    # Bootstrap server not ready, wait for a while and retry.
                    await asyncio.sleep(1)
                except Exception as e:
                    err_msg = (
                        e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                    )
                    logger.error(
                        "Error registering %s with bootstrap server: %s", payload, err_msg
                    )
                    _producer_trace(
                        "bootstrap_register_error tp_rank=%d err=%s",
                        vtp_rank,
                        err_msg,
                    )
                    raise e

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        _producer_trace(
            "sender_listener_bound host=%s side_channel_port=%d",
            self.hostname,
            self.side_channel_port,
        )
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )

        await self.register_worker_with_bootstrap()

        # Create async worker tasks that process items from the queue
        sender_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_sender_tasks)
        ]

        ready_event.set()
        _producer_trace(
            "sender_listener_ready host=%s side_channel_port=%d sender_tasks=%d",
            self.hostname,
            self.side_channel_port,
            self.num_sender_tasks,
        )

        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                await self.sender_worker_queue.put((identity, metadata_bytes))
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            # Clean up worker tasks
            for task in sender_tasks:
                task.cancel()
            await asyncio.gather(*sender_tasks, return_exceptions=True)
            sock.close()

    async def _sender_worker(self, sock: zmq.asyncio.Socket):
        while True:
            try:
                identity, metadata_bytes = await self.sender_worker_queue.get()
                try:
                    metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                    logger.debug(
                        "Mooncake side-channel received: reqs=%s ack=%s "
                        "remote=%s:%s remote_tp_size=%s remote_tp_rank=%s",
                        list(metadata.req_blocks),
                        metadata.ack_for_transfers,
                        metadata.remote_hostname,
                        metadata.remote_port,
                        metadata.remote_tp_size,
                        metadata.remote_tp_rank,
                    )
                    if metadata.ack_for_transfers:
                        _producer_trace(
                            "ack_recv transfers=%s remote_tp_rank=%s",
                            metadata.ack_for_transfers,
                            metadata.remote_tp_rank,
                        )
                    else:
                        _producer_trace(
                            "pull_req_recv reqs=%s remote=%s:%s remote_tp_rank=%s "
                            "remote_tp_size=%s",
                            list(metadata.req_blocks),
                            metadata.remote_hostname,
                            metadata.remote_port,
                            metadata.remote_tp_rank,
                            metadata.remote_tp_size,
                        )
                    # Layer 2' (READ pull): ACK from consumer after sync_read done
                    if metadata.ack_for_transfers:
                        self._handle_read_ack(
                            metadata.ack_for_transfers, metadata.remote_tp_rank
                        )
                        # Send ACK_OK handshake reply so consumer knows we got it
                        ack_ok = MooncakeXferResponse(
                            status=MooncakeXferResponseStatus.ACK_OK,
                        )
                        await sock.send_multipart(
                            (identity, self._encoder.encode(ack_ok))
                        )
                        logger.warning(
                            "Mooncake side-channel sent ACK_OK: transfers=%s",
                            metadata.ack_for_transfers,
                        )
                    else:
                        await self.send_kv_to_decode(identity, sock, metadata)
                except Exception as e:
                    logger.error("Error processing Mooncake xfer request: %s", e)
                    error_response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR, err_msg=str(e)
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(error_response))
                    )
                finally:
                    self.sender_worker_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in _sender_worker: %s", e)

    def _handle_read_ack(
        self, transfer_ids: list[TransferId], remote_tp_rank: int
    ):
        """Layer 2' (READ pull): consumer ACKed that batch_transfer_sync_read
        completed — now safe to free producer-side blocks."""
        for transfer_id in transfer_ids:
            if transfer_id not in self.reqs_need_send:
                continue
            send_meta = self.reqs_need_send[transfer_id]
            if send_meta.acked_remote_tp_ranks is None:
                send_meta.acked_remote_tp_ranks = set()
            if remote_tp_rank in send_meta.acked_remote_tp_ranks:
                continue
            send_meta.acked_remote_tp_ranks.add(remote_tp_rank)
            send_meta.sending = max(0, send_meta.sending - 1)
            send_meta.sent += 1
            _producer_trace(
                "ack_apply transfer=%s p_req=%s remote_tp_rank=%s "
                "sent=%d need_send=%d sending=%d",
                transfer_id,
                send_meta.p_req_id,
                remote_tp_rank,
                send_meta.sent,
                send_meta.need_send,
                send_meta.sending,
            )
            if send_meta.sent == send_meta.need_send:
                del self.reqs_need_send[transfer_id]
                self.finished_sending_reqs.add(send_meta.p_req_id)
                _producer_trace(
                    "ack_finish transfer=%s p_req=%s",
                    transfer_id,
                    send_meta.p_req_id,
                )

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks(meta.remote_tp_size)
        if self.tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            msg = f"This P tp_rank {self.tp_rank} not in remote D target ranks {remote_tp_ranks}"  # noqa: E501
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
                self.reqs_need_send[transfer_id] = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
                logger.debug(
                    "Mooncake side-channel created pending send placeholder: "
                    "d_req=%s transfer=%s",
                    d_req_id,
                    transfer_id,
                )
            send_meta = self.reqs_need_send[transfer_id]
            pending_reqs[d_req_id] = send_meta
            logger.warning(
                "Mooncake side-channel waiting for ready: d_req=%s "
                "transfer=%s ready=%s local_blocks=%d p_req=%s",
                d_req_id,
                transfer_id,
                send_meta.ready.is_set(),
                len(send_meta.local_block_ids),
                send_meta.p_req_id,
            )
            _producer_trace(
                "pull_wait_ready d_req=%s transfer=%s ready=%s local_blocks=%d p_req=%s",
                d_req_id,
                transfer_id,
                send_meta.ready.is_set(),
                len(send_meta.local_block_ids),
                send_meta.p_req_id,
            )

        async def wait_and_ret(
            d_req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Timeout, abort all pending requests.
                for task in wait_tasks:
                    task.cancel()
                logger.warning(
                    "Timeout waiting for P side ready: %s", list(pending_reqs)
                )
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    err_reqs=list(pending_reqs),
                    err_msg="Timeout waiting for P side ready.",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                break

            wait_tasks = list(pending)
            response_status = (
                MooncakeXferResponseStatus.CONTINUE
                if wait_tasks
                else MooncakeXferResponseStatus.FINISH
            )
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done:
                d_req_id, send_meta = task.result()
                del pending_reqs[d_req_id]
                # Do we still in reqs_need_send (not expired)?
                if send_meta.transfer_id in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta, remote_tp_ranks)
                    ready_reqs.append((d_req_id, send_meta))
                    _producer_trace(
                        "pull_ready d_req=%s transfer=%s p_req=%s blocks=%d need_send=%d",
                        d_req_id,
                        send_meta.transfer_id,
                        send_meta.p_req_id,
                        len(send_meta.local_block_ids),
                        send_meta.need_send,
                    )
                else:
                    # Otherwise (expired, very unlikely), just forget it.
                    logger.warning(
                        "Request %s expired before sending on P side.", d_req_id
                    )

            max_reqs_per_response = int(
                os.environ.get("VLLM_MOONCAKE_MAX_REQS_PER_RESPONSE", "8")
            )
            ready_req_chunks = self._chunk_ready_reqs(
                ready_reqs, max_reqs_per_response
            )

            # Layer 2' (READ pull) env gate
            use_read_pull = os.environ.get(
                "VLLM_MOONCAKE_USE_READ_PULL", "0"
            ) == "1"

            for chunk_idx, ready_req_chunk in enumerate(ready_req_chunks):
                chunk_status = (
                    response_status
                    if chunk_idx == len(ready_req_chunks) - 1
                    else MooncakeXferResponseStatus.CONTINUE
                )
                src_ptrs, dst_ptrs, lengths, err_reqs = (
                    await self._build_transfer_params(ready_req_chunk, meta)
                )
                logger.warning(
                    "Mooncake side-channel built transfer params: ok_reqs=%s "
                    "src_ptrs=%d dst_ptrs=%d lengths=%d err_reqs=%s "
                    "chunk=%d/%d max_reqs_per_response=%d",
                    [d_req_id for d_req_id, _ in ready_req_chunk],
                    len(src_ptrs),
                    len(dst_ptrs),
                    len(lengths),
                    err_reqs,
                    chunk_idx + 1,
                    len(ready_req_chunks),
                    max_reqs_per_response,
                )
                _producer_trace(
                    "pull_chunk reqs=%s chunk=%d/%d src_ptrs=%d lengths=%d err=%s",
                    [d_req_id for d_req_id, _ in ready_req_chunk],
                    chunk_idx + 1,
                    len(ready_req_chunks),
                    len(src_ptrs),
                    len(lengths),
                    err_reqs,
                )

                if err_reqs:
                    response = MooncakeXferResponse(
                        status=chunk_status,
                        err_reqs=err_reqs,
                        err_msg="P num blocks less than D",
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(response))
                    )

                if src_ptrs and not use_read_pull:
                    # Original WRITE push path
                    remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
                    ret_value = await self.sender_loop.run_in_executor(
                        self._sender_executor,
                        self._send_blocks,
                        remote_session,
                        src_ptrs,
                        dst_ptrs,
                        lengths,
                    )

                    if ret_value != 0:
                        err_reqs = []
                        for d_req_id, send_meta in ready_req_chunk:
                            send_meta.sending -= 1
                            err_reqs.append(d_req_id)
                        # Do best effort to transfer the remaining reqs.
                        response = MooncakeXferResponse(
                            status=chunk_status,
                            err_reqs=err_reqs,
                            err_msg=f"Mooncake transfer engine returned {ret_value}",
                        )
                        await sock.send_multipart(
                            (identity, self._encoder.encode(response))
                        )
                        continue

                if not use_read_pull:
                    # WRITE push: transfer done, mark sent immediately.
                    for d_req_id, send_meta in ready_req_chunk:
                        send_meta.sending -= 1
                        send_meta.sent += 1
                        if send_meta.sent == send_meta.need_send:
                            del self.reqs_need_send[send_meta.transfer_id]
                            self.finished_sending_reqs.add(send_meta.p_req_id)

                    response = MooncakeXferResponse(
                        status=chunk_status,
                        ok_reqs=[d_req_id for d_req_id, _ in ready_req_chunk],
                    )
                else:
                    # READ pull: include src/dst metadata; defer sent/finished
                    # until consumer sends ACK (via _handle_read_ack).
                    p_session = (
                        f"{self.hostname}:{self.rpc_port}" if src_ptrs else None
                    )
                    response = MooncakeXferResponse(
                        status=chunk_status,
                        ok_reqs=[d_req_id for d_req_id, _ in ready_req_chunk],
                        p_session=p_session,
                        p_src_ptrs=src_ptrs if src_ptrs else None,
                        p_dst_ptrs=dst_ptrs if src_ptrs else None,
                        p_lengths=lengths if src_ptrs else None,
                    )

                await sock.send_multipart((identity, self._encoder.encode(response)))
                logger.warning(
                    "Mooncake side-channel sent response: status=%s ok=%s err=%s "
                    "p_session=%s ptrs=%d chunk=%d/%d",
                    response.status.name,
                    response.ok_reqs,
                    response.err_reqs,
                    response.p_session,
                    len(response.p_src_ptrs or []),
                    chunk_idx + 1,
                    len(ready_req_chunks),
                )
                _producer_trace(
                    "pull_reply status=%s ok=%s err=%s ptrs=%d chunk=%d/%d",
                    response.status.name,
                    response.ok_reqs,
                    response.err_reqs,
                    len(response.p_src_ptrs or []),
                    chunk_idx + 1,
                    len(ready_req_chunks),
                )

    def resolve_need_send(self, send_meta: SendBlockMeta, remote_tp_ranks: list[int]):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        num_virtual_tp = int(os.environ.get("VLLM_MOONCAKE_VIRTUAL_TP_SIZE", "0"))
        if send_meta.need_send != 1 and num_virtual_tp == 0:
            logger.error("Mooncake: Heterogeneous TP is not supported yet.")
            raise NotImplementedError(
                "Mooncake: Heterogeneous TP is not supported yet."
            )

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
    ) -> tuple[list[int], list[int], list[int], list[ReqId]]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        err_reqs: list[ReqId] = []
        local_base_addr = self.kv_caches_base_addr
        remote_base_addr = agent_meta.kv_caches_base_addr
        block_len = self.block_len
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids = agent_meta.req_blocks[d_req_id]
            num_remote_blocks = len(remote_block_ids)
            if num_remote_blocks == 0:
                continue

            local_block_ids = send_meta.local_block_ids
            # Partial prefix cache hit: just read uncomputed blocks.
            num_local_blocks = len(local_block_ids)
            if num_local_blocks < num_remote_blocks:
                logger.error(
                    "req %s: local blocks(%d) less than remote blocks(%d)!",
                    d_req_id,
                    num_local_blocks,
                    num_remote_blocks,
                )
                err_reqs.append(d_req_id)
                continue
            if num_local_blocks > num_remote_blocks:
                local_block_ids = local_block_ids[-num_remote_blocks:]

            # Group by indices
            group_local_block_ids, group_remote_block_ids = group_concurrent_contiguous(
                local_block_ids, remote_block_ids
            )

            for local_layer_addr, remote_layer_addr in zip(
                local_base_addr, remote_base_addr
            ):
                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    src_ptrs.append(
                        local_layer_addr + group_local_block_id[0] * block_len
                    )
                    dst_ptrs.append(
                        remote_layer_addr + group_remote_block_id[0] * block_len
                    )
                    lengths.append(block_len * len(group_local_block_id))

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                d_req_id,
                num_remote_blocks,
                remote_session,
            )

        return src_ptrs, dst_ptrs, lengths, err_reqs

    @staticmethod
    def _chunk_ready_reqs(
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        max_reqs_per_response: int,
    ) -> list[list[tuple[ReqId, SendBlockMeta]]]:
        if max_reqs_per_response <= 0 or len(ready_reqs) <= max_reqs_per_response:
            return [ready_reqs]

        return [
            ready_reqs[i:i + max_reqs_per_response]
            for i in range(0, len(ready_reqs), max_reqs_per_response)
        ]

    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        if ret_value == 0:
            logger.debug(
                "Sending to %s done, took %s",
                remote_session,
                time.perf_counter() - start_time,
            )
        return ret_value

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs = []
        kv_data_lens = []
        seen_base_addresses = []

        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            logger.debug(
                "registering layer %s with shape %s", layer_name, cache_or_caches.shape
            )
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)
                curr_tensor_size_bytes = cache.nbytes

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]

                assert tensor_size_bytes == curr_tensor_size_bytes, (
                    "All kv cache tensors must have the same size"
                )
                kernel_block_size = cache.shape[-2 if self.use_mla else -3]
                assert self.block_size == kernel_block_size
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(tensor_size_bytes)

        self.kv_caches_base_addr = seen_base_addresses

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0
        self.block_len = tensor_size_bytes // self.num_blocks
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_len=%d", self.num_blocks, self.block_len
        )

        # No need to launch server for D node.
        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        listener_future = asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
        )
        if not ready_event.wait(timeout=60):
            if listener_future.done():
                listener_exc = listener_future.exception()
                if listener_exc is not None:
                    raise RuntimeError(
                        "Mooncake sender listener failed during startup"
                    ) from listener_exc
            raise RuntimeError(
                "Timed out waiting for Mooncake sender listener startup."
            )
        if listener_future.done():
            listener_exc = listener_future.exception()
            if listener_exc is not None:
                raise RuntimeError(
                    "Mooncake sender listener exited unexpectedly during startup"
                ) from listener_exc

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        finished_recving_reqs = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        return finished_recving_reqs

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()

        expired_transfer_id = []
        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                finished_sending_reqs.add(send_meta.p_req_id)
                expired_transfer_id.append(transfer_id)

        for transfer_id in expired_transfer_id:
            del self.reqs_need_send[transfer_id]

        return finished_sending_reqs

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        recv_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if not self.is_kv_consumer:
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )

        finished_recving_reqs = recv_fut.result() if recv_fut else set()
        finished_sending_reqs = send_fut.result() if send_fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        return finished_sending_reqs or None, finished_recving_reqs or None

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        req_ids = set(pull_metas)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug(
            "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
        )

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(encoded_data)
                pending_responses: list[MooncakeXferResponse] = []
                use_read_pull = os.environ.get(
                    "VLLM_MOONCAKE_USE_READ_PULL", "0"
                ) == "1"
                while True:
                    if pending_responses:
                        response = pending_responses.pop(0)
                    else:
                        ret_msg = await sock.recv()
                        response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        logger.error(
                            "Error happens during tranfering kvcache for %s: %s",
                            req_ids,
                            response.err_msg,
                        )
                        return
                    if response.status == MooncakeXferResponseStatus.ACK_OK:
                        logger.warning(
                            "Unexpected ACK_OK while receiving KV for %s", req_ids
                        )
                        continue
                    # Layer 2' (READ pull): if producer sent src metadata, we
                    # perform the sync_read locally. batch_transfer_sync_read
                    # returns only after cudaMemcpy H2D completes, so the KV
                    # is guaranteed to be in consumer GPU memory before we
                    # proceed to mark finished_recving_reqs.
                    if response.p_session and response.p_src_ptrs:
                        loop = asyncio.get_running_loop()
                        ret_value = await loop.run_in_executor(
                            None,
                            self.engine.batch_transfer_sync_read,
                            response.p_session,
                            response.p_dst_ptrs,
                            response.p_src_ptrs,
                            response.p_lengths,
                        )
                        if ret_value != 0:
                            logger.error(
                                "batch_transfer_sync_read failed for %s: %d",
                                req_ids,
                                ret_value,
                            )
                            # Don't send ACK on failure; let producer timeout.
                            return

                    # Layer 2' reliability (review fixes):
                    # - ACK per response (not only FINISH) so CONTINUE batches
                    #   are also acked; avoids strand if later FINISH fails.
                    # - Send ACK even when p_session was None (src_ptrs == [])
                    #   so producer always finalizes send_meta.sent.
                    # - Wait for ACK_OK reply from producer before proceeding;
                    #   guarantees actual delivery, not just local ZMQ queue.
                    if use_read_pull and response.ok_reqs:
                        ack_transfer_ids = list(
                            {
                                pull_metas[r].transfer_id
                                for r in response.ok_reqs
                                if r in pull_metas
                            }
                        )
                        if ack_transfer_ids:
                            ack_meta = MooncakeXferMetadata(
                                remote_hostname="",
                                remote_port=0,
                                remote_tp_size=0,
                                remote_tp_rank=self.tp_rank,
                                req_blocks={},
                                kv_caches_base_addr=[],
                                ack_for_transfers=ack_transfer_ids,
                            )
                            await sock.send(self._encoder.encode(ack_meta))
                            # Wait for producer's ACK_OK handshake reply.
                            try:
                                while True:
                                    ack_msg = await sock.recv()
                                    ack_response = self._xfer_resp_decoder.decode(
                                        ack_msg
                                    )
                                    if (
                                        ack_response.status
                                        == MooncakeXferResponseStatus.ACK_OK
                                    ):
                                        break
                                    pending_responses.append(ack_response)
                            except Exception as e:
                                logger.error(
                                    "ACK_OK recv failed for %s: %s", req_ids, e
                                )
                                return

                    self.process_pulling_result(response, pull_metas)

                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            return

    def process_pulling_result(
        self,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        ok_reqs: list[ReqId] = response.ok_reqs or []

        for req_id in ok_reqs:
            pull_meta = pull_metas[req_id]
            # No race because we are in async loop.
            pull_meta.pull_tasks_count -= 1
            if pull_meta.pull_tasks_count == 0:
                self.finished_recving_reqs.add(pull_meta.d_req_id)

        if ok_reqs:
            logger.debug("pulling kv_caches for %s finished", ok_reqs)

        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data: dict = response.json()
                for _, dp_entry in data.items():
                    remote_engine_id = dp_entry["engine_id"]
                    self._remote_agents[remote_engine_id] = {
                        int(tp_rank): {
                            int(pp_rank): worker_addr
                            for pp_rank, worker_addr in tp_entry.items()
                        }
                        for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                    }
                    self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
        except Exception as e:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                e,
            )

        # Always notify others regardless of connection success or failure.
        self._pending_bootstrap_querys[remote_bootstrap_addr].set()
        del self._pending_bootstrap_querys[remote_bootstrap_addr]

    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks_from_engine_id(
            remote_engine_id
        )
        count = len(remote_tp_ranks)
        if count != 1:
            logger.error("Mooncake: Heterogeneous TP is not supported yet.")
            raise NotImplementedError(
                "Mooncake: Heterogeneous TP is not supported yet."
            )
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
        for remote_tp_rank in remote_tp_ranks:
            worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][0]
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )

    async def handle_new_engine_id(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
        if remote_bootstrap_addr not in self._pending_bootstrap_querys:
            self._pending_bootstrap_querys[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_querys[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            return

        self.receive_kv(remote_engine_id, pull_metas)

    async def _start_load_kv(
        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
    ):
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
            else:
                self.receive_kv(remote_engine_id, pull_metas)

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if block_ids:
                # Already gone through request_finished()
                if transfer_id not in self.reqs_need_send:
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
                    logger.warning(
                        "Mooncake producer created missing send placeholder at "
                        "ready time: p_req=%s transfer=%s",
                        p_req_id,
                        transfer_id,
                    )
                send_meta = self.reqs_need_send[transfer_id]
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                send_meta.ready.set()
                _producer_trace(
                    "send_ready req=%s transfer=%s blocks=%d expire_in=%.1fs",
                    p_req_id,
                    transfer_id,
                    len(block_ids),
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                logger.debug(
                    "Mooncake producer marked send ready: p_req=%s transfer=%s "
                    "blocks=%d pending=%d",
                    p_req_id,
                    transfer_id,
                    len(block_ids),
                    len(self.reqs_need_send),
                )
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
                # when D is sending MooncakeXferMetadata.
                if transfer_id not in self.reqs_need_send:
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
                    logger.debug(
                        "Mooncake producer registered prefill placeholder: "
                        "p_req=%s transfer=%s pending=%d",
                        p_req_id,
                        transfer_id,
                        len(self.reqs_need_send),
                    )
        for transfer_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(transfer_id, None)
            if send_meta:
                assert not send_meta.ready.is_set()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if not self.is_kv_producer and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if not self.is_kv_consumer and (
            metadata.reqs_to_send or metadata.reqs_not_processed
        ):
            asyncio.run_coroutine_threadsafe(
                self.record_send_reqs(metadata), self.sender_loop
            )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_index
        * vllm_config.parallel_config.tensor_parallel_size
    )


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool:
    assert (parallel_config := vllm_config.parallel_config)
    # In hybrid or external LB mode,
    # each instance should have its own bootstrap server.
    #
    # In internal LB mode,
    # only the real global first rank need to launch the bootstrap server.
    return is_local_first_rank() and (
        parallel_config.local_engines_only or parallel_config.data_parallel_index == 0
    )


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers.
    Decoders should get addr from kv_transfer_params.
    """
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        # In hybrid or external LB mode, connect to local server.
        host = "127.0.0.1"
    else:
        host = parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)
