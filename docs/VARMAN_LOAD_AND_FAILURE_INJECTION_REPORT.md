# Varman 4.0 — Load & Failure-Injection Verification Report

This document reports the empirical validation of Varman 4.0's concurrency controls, queue boundaries, and resilience under simulated failure scenarios.

---

## 1. Concurrency & Queue-Boundary Testing (1 to 100 Clients)

> [!IMPORTANT]
> **Test Classification:**  
> The 1-to-100 concurrent test is explicitly classified as a **queue-boundary enforcement test**, NOT a capacity or throughput measurement.  
> Its sole objective is to verify that the application strictly rejects requests beyond its bounded queue capacity, preventing memory exhaustion, unmanaged thread creation, or PyTorch CUDA context corruption.

### Queue Architecture & Boundary Dynamics
- **Executing Worker Semaphore:** 1 active worker thread (`asyncio.Semaphore(1)`).
- **Maximum Pending Queue Depth:** 5 pending jobs (`MAX_PENDING_QUEUE = 5`).
- **Admitted Task Capacity:** Between 5 and 6 total requests in-flight.
  - *Instantaneous Burst Arrival:* If requests arrive simultaneously before the executing thread acquires the semaphore and decrements the queue counter, exactly **5** are admitted and request 6 is rejected with `HTTP 429`.
  - *Interleaved Worker Arrival:* If task 1 transitions from pending to processing and decrements `_pending_jobs_count` before request 6 arrives, a 6th request is admitted to fill the vacated pending slot (yielding **6** total in-flight: 1 executing + 5 pending).
- **Rejection Status:** `HTTP 429 Too Many Requests` with header `Retry-After: 30`.

### Test Matrix Results (`tests/test_load_concurrency_queue.py`)

All concurrent requests were dispatched simultaneously against `/api/images/protect` under mock pipeline execution:

| Concurrent Clients | Total In-Flight Admitted | Queue Rejections (HTTP 429) | Rejection % | Boundary Behavior |
| :---: | :---: | :---: | :---: | :--- |
| **1** | 1 | 0 | 0.0% | 1 executing immediately, 0 pending |
| **5** | 5 | 0 | 0.0% | 1 executing, 4 pending |
| **10** | 5 to 6 | 4 to 5 | 40%–50% | Queue saturated; excess rejected with 429 |
| **25** | 5 to 6 | 19 to 20 | 76%–80% | Queue saturated; excess rejected with 429 |
| **50** | 5 to 6 | 44 to 45 | 88%–90% | Queue saturated; excess rejected with 429 |
| **100** | 5 to 6 | 94 to 95 | 94%–95% | Queue saturated; excess rejected with 429 |

---

## 2. Failure-Injection Testing

### Scenario 1: CPU-Only Server Host with Hard Mode Request
- **Injected Fault:** Server has no CUDA GPU (`probe_gpu_available() -> False`).
- **Input:** Request specifies `mode="hard"`, `allow_cpu_fallback=False`.
- **Observed Behavior:**
  - Backend immediately aborts processing with `HTTP 409 Conflict`.
  - Zero disk space consumed (upload unlinked).
  - Zero queue slots allocated.
  - Returns clear instruction: explicit consent `allow_cpu_fallback=true` required.
- **Verdict:** **PASSED** (Validated in `test_hard_mode_on_cpu_server_without_consent_returns_409`).

### Scenario 2: Sudden Process Termination / Server Restart
- **Injected Fault:** Server process terminated while jobs are in `pending` or `processing` states.
- **Recovery Mechanism:** Lifespan startup handler executes:
  ```python
  stmt = update(ProtectedImage).where(
      ProtectedImage.status.in_(["pending", "processing"])
  ).values(status="failed", error_message="Interrupted by server restart")
  ```
- **Observed Behavior:** On next startup, all stale jobs are marked `failed`. No jobs hang indefinitely in `processing`.
- **Verdict:** **PASSED**.

### Scenario 3: Cooperative In-Flight Job Cancellation
- **Injected Fault:** User triggers `POST /api/images/cancel/{image_id}` while PGD optimization is active at iteration $t = 12 / 25$.
- **Observed Behavior:**
  - `_cancel_events[image_id].set()` is triggered.
  - The step loop in `PasteAwareProtectionEngine` detects the event, aborts computation, releases CUDA tensors, and raises `TaskCancelledException`.
  - Database record status transitions to `cancelled`.
  - Partial unperturbed files are unlinked from disk.
- **Verdict:** **PASSED** (Validated in `tests/test_cooperative_cancellation.py`).

### Scenario 4: Malicious & Invalid Presets Injection
- **Injected Fault:** Attacker sends `mode="quantum"` or `preset_name="TurboExtreme"`.
- **Observed Behavior:**
  - Backend returns `HTTP 422 Unprocessable Entity`.
  - No background task is spawned; queue remains untouched.
- **Verdict:** **PASSED** (Validated in `test_validation_rejects_unknown_mode_422`).

---

## 3. Scope Boundaries & Untested Failure Modes

> [!WARNING]
> **Operational Caveats & Untested Conditions:**
> 1. **VRAM Driver Creep:** Synthetic unit tests execute with mock pipelines; actual driver memory stabilization across sequential real-world runs requires manual validation (e.g., Step 9 via `nvidia-smi` across 5+ jobs).
> 2. **Missing Weights:** Pipeline behavior if surrogate weight files (`weights/model18.onnx` or Facenet weights) are corrupted or deleted at runtime is untested.
> 3. **GPU Out-of-Memory (OOM):** Handling of multi-face high-resolution uploads (>3 simultaneous faces on high-res crops) on hosts with <4GB VRAM has not been stressed.
> 4. **Worker Timeouts:** Long-duration worker timeouts (e.g. 45s limit) have been unit-tested via asyncio timeout handlers, but real sustained saturation under multi-gigabyte loads remains unverified.
> 5. **High-Concurrency Duplicate Submissions:** Concurrent race conditions where the identical payload is submitted simultaneously by multiple sessions are handled by the database idempotency query, but concurrent write collisions have not been formally load-tested.
