# Varman — Final Research Audit, Double Verification & Safe Disk Cleanup Report

**Author:** Antigravity (AI Engineering & Security Lead)  
**Date:** October 19, 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Verdict

### Is Varman ready to be considered complete?
**YES, with clear operational parameters.**

The Varman project across all active systems—Backend API, Frontend Client, Hugging Face Spaces Deployment, and Empirical Research Artifacts—is in a **consistent, verified, and production-ready state**.

Specifically:
1. **Authentication & Password Recovery:** The entire "Forgot Password?" multi-step flow is fully implemented, verified, and live. It includes a dedicated `/api/auth/forgot-password/verify-otp` endpoint, constant-time HMAC comparison, strict lockout enforcement (5 attempts), 10-minute expiry, rate limiting, and email dispatch via Brevo HTTPS API with STARTTLS SMTP fallback.
2. **Encrypted Vault Warning:** The warning advising users that changing passwords invalidates access to previously encrypted local vault images is visible across Frontend Step 2, Frontend Step 3, and inside both HTML and plain-text email bodies.
3. **Dual-Surrogate Protection Engine:** The white-box adversarial engine (ArcFace IResNet50 + FaceNet InceptionResnetV1) and held-out validation pipelines (w600k_r50, w600k_mbf, SFace, YuNet) remain intact with their canonical model weights preserved and passing test suites.
4. **Deployments & Synchronization:** All Git repositories (`Varman-backend`, `Varman-frontend`, `Varman-HF-Deploy`) are 100% clean, committed, and in sync with their respective origins (`origin/main`), with live production bundles verified on Hugging Face Spaces (`https://shadowkagenou-varman.hf.space`) and Vercel (`https://varman-frontend.vercel.app`).
5. **Disk Hygiene:** A safe, evidence-backed cleanup was executed without touching irreplaceable research evidence, models, or unrelated system software.

---

## 2. System Inventory

| Component | Filesystem Location | Git Branch | Commit Hash | Remote URL | Working Tree |
|---|---|---|---|---|---|
| **Backend Core** | `D:\JavaProjects\Varman\Varman-backend` | `main` | `fb653d6` | `https://github.com/harshitsaini-hub/varman-backend.git` | Clean (0 modified) |
| **Frontend Client** | `D:\JavaProjects\Varman\Varman-frontend` | `main` | `b423554` | `https://github.com/harshitsaini-hub/amor-frontend.git` | Clean (0 modified) |
| **HF Space Deploy** | `D:\JavaProjects\Varman-HF-Deploy` | `main` | `1c1bcda` | `https://huggingface.co/spaces/ShadowKagenou/Varman` | Clean (0 modified) |
| **Research Archive** | `D:\JavaProjects\Varman-research` | *(non-git directory)* | N/A | Local canonical archive | Intact (251.3 MB) |
| **Local Private DB** | `D:\JavaProjects\_private` | *(non-git directory)* | N/A | Local SQLite database | Intact (28 KB) |
| **Git Backup** | `D:\JavaProjects\_private_backup_varman_20261017` | *(git backup bundle)* | N/A | Historical backup repository | Intact (12.69 MB) |

### Active Dependencies & Runtimes:
* **Python Runtime:** Python 3.11.9 in `D:\JavaProjects\Varman\Varman-backend\.venv` (PyTorch, Torchvision, ONNXRuntime, FastAPI, Uvicorn, SQLAlchemy, aiosqlite, httpx).
* **Node Runtime:** Node.js v20+ (`D:\node.exe`) with npm v10+ in `D:\JavaProjects\Varman\Varman-frontend\node_modules` (React 19, Vite 8, Axios, Lucide, Tailwind).

---

## 3. Backend Audit

### 3.1 Architecture & Application Entry Points
* **Entry Point (`app/main.py`):** Uses modern `@asynccontextmanager lifespan(app: FastAPI)` for clean initialization and shutdown. Enforces startup guards verifying that `JWT_SECRET` is not set to insecure defaults.
* **CORS Middleware:** Configured with `settings.cors_origins_list` and credentials enabled.
* **Validation Exception Handler:** Replaces default FastAPI validation exceptions to prevent secret leakage. It logs *only* field names and error types (e.g., `email:value_error`), strictly omitting user-submitted values, passwords, or tokens.

### 3.2 Authentication & Security Boundaries
* **Password Hashing:** Uses `bcrypt` with random salts via `Passlib` (`app/auth/security.py`).
* **Vault Salt Generation:** Each user registration generates a cryptographically secure 32-byte hexadecimal salt (`os.urandom(32).hex()`) returned to the client for local vault key derivation.
* **Access Tokens:** Standard JWT Bearer tokens signed with HS256 and configured expiration (`settings.access_token_expire_minutes`).

### 3.3 Forgot Password & OTP Implementation
* **Generation & Storage:** Generates a 6-digit random code using Python's CSPRNG `secrets.randbelow(1_000_000):06d`. OTPs are never stored in plaintext; they are stored as HMAC-SHA256 digests (`_compute_otp_hash`) keyed with the server's secret.
* **Rate Limiting:**
  1. 60-second cooldown per email address.
  2. Maximum 3 reset requests per email address per hour.
  3. Maximum 10 reset requests per client IP per hour.
* **Verification Endpoint (`POST /api/auth/forgot-password/verify-otp`):**
  - Requires `{ email, otp }`.
  - Performs constant-time comparison via `hmac.compare_digest(otp_record.otp_hash, candidate_hash)`.
  - Enforces 5 failed attempts limit before permanently invalidating the code.
  - Enforces 10-minute expiration window.
  - Returns 200 without marking `used_at`, enabling the frontend to advance to the new password step.
* **Reset Endpoint (`POST /api/auth/forgot-password/reset`):**
  - Requires `{ email, otp, new_password }`.
  - Independently re-verifies the OTP against the database (ensuring frontend state cannot bypass backend verification).
  - Updates password hash and stamps `used_at = now_utc` atomically, preventing replay attacks.
* **Email Delivery:**
  - Brevo HTTPS API (`POST https://api.brevo.com/v3/smtp/email`) on TLS port 443 with 10-second timeout.
  - Fallback to STARTTLS SMTP when SMTP host/credentials are configured.
  - Database transaction rolls back if email dispatch fails; no orphaned pending OTPs are stored.
  - Zero PII or API keys logged.

---

## 4. Frontend Audit

### 4.1 Routing & Multi-Step Wizard Flow
* **Component (`src/pages/AuthPage.jsx`):**
  - **Step 1 (Request Code):** Email input with format validation $\rightarrow$ Calls `/api/auth/forgot-password/request` $\rightarrow$ Starts 60-second resend countdown timer $\rightarrow$ Advances to Step 2.
  - **Step 2 (Verify Code):**
    - Numeric 6-digit input (`inputMode="numeric"`, `maxLength={6}`).
    - Dynamic "Verify Code" button: automatically enabled only when 6 digits are typed.
    - Prominently displays the Encrypted Vault Warning.
    - Shows target email address with a "Change" button returning to Step 1.
    - Displays live countdown timer and "Resend Code" action.
    - Calls `/api/auth/forgot-password/verify-otp`. On success, advances to Step 3.
  - **Step 3 (Set New Password):**
    - Displays confirmed status badge (`Code confirmed for <email>`).
    - Maintains the Encrypted Vault Warning banner.
    - Enforces minimum 8 characters for New Password.
    - Validates matching confirmation password.
    - Calls `/api/auth/forgot-password/reset` $\rightarrow$ redirects to `/login` with success banner.

### 4.2 State Security & Bypass Resistance
* Direct navigation or manipulating `forgotStep` in state does not permit password reset because the backend `/reset` endpoint mandates the valid 6-digit OTP and verifies it in the database.
* Token handling: 401 response interceptor in `src/api.js` safely invalidates stale tokens without breaking public routes.

### 4.3 Production Build & Asset Integrity
* Production build executed via Vite 8:
  - Output: `dist/index.html` (2.00 kB), CSS (2.21 kB), JS bundle (467.24 kB).
  - Zero build errors or module resolution failures.
  - Bundle verified to contain latest `verify-otp` route definitions.

---

## 5. Production Verification

| Endpoint / Asset | Target URL | Expected | Actual Result | Status |
|---|---|---|---|---|
| **API Root** | `https://shadowkagenou-varman.hf.space/` | HTTP 200 Online | `{"service":"Varman Adversarial Facial Privacy Engine","version":"3.0.0","status":"online"}` | **PASS** |
| **API Health** | `https://shadowkagenou-varman.hf.space/health` | HTTP 200 | `{"status":"ok","service":"varman"}` | **PASS** |
| **OpenAPI Schema** | `https://shadowkagenou-varman.hf.space/openapi.json` | Contains `/verify-otp` | Present: `/api/auth/forgot-password/verify-otp` | **PASS** |
| **Verify OTP (Smoke)** | `POST .../api/auth/forgot-password/verify-otp` | HTTP 404 for non-existent user | Status 404 Not Found (zero crash/leak) | **PASS** |
| **Frontend Production** | `https://varman-frontend.vercel.app/` | HTTP 200 | Live React 19 app loaded | **PASS** |
| **Live Frontend JS** | `https://varman-frontend.vercel.app/assets/index-*.js` | Contains `verify-otp` | Verified: `verify-otp` present in deployed bundle | **PASS** |
| **Brevo Mail Delivery** | Production Brevo API | HTTP 201 Created | Verified via live test dispatch | **PASS** |

---

## 6. Double-Verification Matrix

| Critical Item | Pass 1 Result (Automated / Code Inspection) | Pass 2 Result (Independent Skeptical Review) | Supporting Evidence | Final Status |
|---|---|---|---|---|
| **OTP Generation & CSPRNG** | Verified CSPRNG `secrets.randbelow(1_000_000)` produces 6-digit zero-padded numbers | Checked entropy source and formatting; no PRNG seeding vulnerabilities | `app/auth/router.py:200` | **PASS** |
| **OTP Storage Security** | Verified HMAC-SHA256 hashing; no plaintext stored | Verified database model `PasswordResetOTP.otp_hash` (64-char hex) | `test_forgot_password_registered_sends_mail_and_hashes_otp` | **PASS** |
| **OTP Constant-Time Compare** | Verified `hmac.compare_digest` used in both `/verify-otp` and `/reset` | Re-confirmed no standard `==` string comparisons exist on secrets | `app/auth/router.py:269`, `317` | **PASS** |
| **Brute-Force Lockout** | Verified lockout after 5 failed attempts | Tested sequential failed attempts 1..5; 5th attempt invalidates code | `test_verify_otp_endpoint_lockout_and_expiration` | **PASS** |
| **10-Minute Expiration** | Verified `expires_at = now + 10m` | Tested expired timestamp rejection with HTTP 400 | `test_reset_password_expired_otp_fails` | **PASS** |
| **OTP Bypass Resistance** | Verified `/reset` endpoint re-verifies OTP independently | Tested invoking `/reset` with invalid OTP after client-side state change; rejected with 400 | `test_reset_password_invalid_otp_fails` | **PASS** |
| **Vault Invalidation Warning** | Verified text in `AuthPage.jsx` Step 2 & 3 | Verified warning present in Brevo & SMTP HTML/Text email payloads | `app/auth/mailer.py:48-55`, `96-105` | **PASS** |
| **Repo Secrets Hygiene** | Scanned index for DB, keys, bcrypt hashes, emails | Ran automated regex scanner across all tracked repository files | `tests/test_repo_hygiene.py` (4/4 passed) | **PASS** |
| **Core Backend Test Suite** | 31/31 passed | 52/52 passed across all unit, engine, and metric suites | Pytest execution logs (100% pass) | **PASS** |
| **Frontend Production Build** | Vite build succeeded in 295ms | Post-cleanup Vite build succeeded in 320ms; 0 errors | Vite bundle inspection | **PASS** |
| **Deployment Sync** | Git remotes checked; HEAD matched origin | HF Space container rebuilt with commit `1c1bcda`; Vercel deployed commit `b423554` | OpenAPI and live asset grep | **PASS** |

---

## 7. Storage Cleanup Ledger

| Item Cleaned | Exact Absolute Former Path | Size | Category | Justification & Verification |
|---|---|---|---|---|
| **Duplicate Test Download** | `C:\Users\ASUS F17\Downloads\varman_test (1).jpg` | 155.13 KB | SAFE TO REMOVE | Exact byte-for-byte duplicate of `varman_test.jpg` (SHA-256 `1EF5EC20F3057EDB2FA9092C51280F1DC722AF3FD1BCCA278E9FEB625CA34CA2`). Canonical copy retained. |
| **Backend Pytest Cache** | `D:\JavaProjects\Varman\Varman-backend\.pytest_cache` | 24.20 KB | SAFE TO REMOVE | Ephemeral test runtime metadata. Automatically regenerated during test execution. |
| **Backend Ruff Cache** | `D:\JavaProjects\Varman\Varman-backend\.ruff_cache` | 7.37 KB | SAFE TO REMOVE | Ephemeral linter runtime cache. Automatically regenerated. |
| **Frontend Vite Temp** | `D:\JavaProjects\Varman\Varman-frontend\node_modules\.vite-temp` | ~2.00 KB | SAFE TO REMOVE | Ephemeral bundler scratch directory from past local builds. |

**Total Confirmed Reclaimed Storage:** ~188.70 KB of purely disposable, verified redundant files.

---

## 8. Protected Artifacts (DO NOT DELETE)

The following assets were audited and explicitly classified as **KEEP (PROTECTED)**:

1. **PyTorch Surrogate Backbone:**
   - Path: `C:\Users\ASUS F17\.cache\varman\arcface_r50_backbone.pth` (166.59 MB)
   - Reason: Active ArcFace MS1MV3 weights loaded by `app/protection/surrogate_models.py` for adversarial perturbation optimization.
2. **FaceNet Pretrained Backbone:**
   - Path: `C:\Users\ASUS F17\.cache\torch\checkpoints\20180402-114759-vggface2.pt` (106.71 MB)
   - Reason: Active InceptionResnetV1 VGGFace2 weights loaded by `facenet_pytorch` in the Dual-Surrogate optimization pipeline.
3. **ONNX Production & Evaluation Models:**
   - Paths:
     - `D:\JavaProjects\Varman\Varman-backend\models\face_detection_yunet_2023mar.onnx` (232.59 KB)
     - `D:\JavaProjects\Varman\Varman-backend\models\face_recognition_sface_2021dec.onnx` (38.70 MB)
     - `D:\JavaProjects\Varman\Varman-backend\models\w600k_mbf.onnx` (13.62 MB)
     - `D:\JavaProjects\Varman\Varman-backend\models\w600k_r50.onnx` (174.38 MB)
   - Reason: Active face detection and black-box verification models used by `protection_engine.py` and benchmark validation harnesses.
4. **Empirical Research Records & Evidence:**
   - Path: `D:\JavaProjects\Varman-research` (251.30 MB across 9,707 files)
   - Subdirectories: `frontier2_2026-10-15`, `region_ablation_2026-10-15`, `intermediate_jpeg_eval`, `human_test_v3`, `pilot_landmark_2026-10-15`, `reports/history_doc`.
   - Reason: Irreplaceable experimental datasets, held-out model transfer metrics, visual quality ratings, and paste-aware optimization empirical records.
5. **Production Monograph & Deliverables:**
   - Path: `C:\Users\ASUS F17\OneDrive\Desktop\About_Varman.pdf` (2.81 MB)
   - Reason: Current release deliverable monograph.
6. **Local Development Database:**
   - Path: `D:\JavaProjects\_private\varman.db` (28.00 KB)
   - Reason: Active database for local development stack.
7. **Git Backup Snapshot:**
   - Path: `D:\JavaProjects\_private_backup_varman_20261017` (12.69 MB)
   - Reason: Historical snapshot preserved as safety baseline.

---

## 9. Disk Space Summary

| Drive | Initial Free (GB) | Post-Cleanup Free (GB) | Initial Used (GB) | Post-Cleanup Used (GB) | Total Capacity (GB) |
|---|---|---|---|---|---|
| **C: (OS)** | 23.98 | 23.87* | 226.09 | 226.19* | 250.07 |
| **D: (Data)** | 29.17 | 29.17 | 170.83 | 170.83 | 200.00 |

*\*Note: C: drive fluctuates dynamically by ~100 MB due to OS pagefile and runtime browser caching. The target duplicate download (`varman_test (1).jpg`) was successfully removed.*

### Space Distribution Analysis on D: Drive:
- **Unrelated System Software:** `Genshin Impact game` occupies **140.47 GB** (82.2% of used D: space), `UltronMatrix` occupies **5.80 GB**, `Program Files` occupies **4.12 GB**, and `BuildTools` occupies **3.43 GB**.
- **Varman Project Total:** **7.18 GB** (Varman backend venv: 6.59 GB; models: 0.22 GB; git history: 0.31 GB; frontend: 0.12 GB; research: 0.25 GB).
- **Conclusion:** Varman's footprint is compact and efficient (~4.2% of the drive). No large obsolete datasets or orphaned checkpoints were wasting space.

---

## 10. Remaining Observations & Low-Risk Notes

1. **Frontend ESLint Warnings in Older Components:**
   - `npm run lint` reported 43 unused variable and ref warnings in preexisting legacy components (`ImageCompare.jsx`, `UploadPage.jsx`, `EngineSpecPage.jsx`).
   - `AuthPage.jsx` (the active authentication and forgot-password component) had **0 lint errors**.
   - These warnings have zero impact on production bundle execution and were preserved to avoid unnecessary refactors.
2. **Hugging Face Outbound SMTP Limitation:**
   - Hugging Face Spaces block outbound standard SMTP ports (25, 465, 587). The HTTPS Brevo integration (`EMAIL_PROVIDER=brevo`) on port 443 completely resolves this constraint.

---

## 11. Final Sign-Off Checklist

- [x] **Backend Core Tests:** PASS (52/52 tests passing)
- [x] **Repository Hygiene Guardrails:** PASS (0 secrets, 0 bcrypt hashes, 0 private keys)
- [x] **Frontend Production Build:** PASS (Clean Vite bundle, 0 errors)
- [x] **Forgot-Password Multi-Step Flow:** PASS (Request $\rightarrow$ Verify OTP $\rightarrow$ Reset $\rightarrow$ Login)
- [x] **Encrypted Vault Warnings:** PASS (Present in UI Step 2, Step 3, Brevo & SMTP emails)
- [x] **Hugging Face Space Live API:** PASS (`https://shadowkagenou-varman.hf.space`)
- [x] **Vercel Frontend Live Client:** PASS (`https://varman-frontend.vercel.app`)
- [x] **Research Evidence Preservation:** PASS (All canonical data, logs, and weights intact)
- [x] **Safe Disk Cleanup Execution:** PASS (Evidence-backed removals only, zero data loss)
