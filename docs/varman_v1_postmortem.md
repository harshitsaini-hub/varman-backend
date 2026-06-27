# Varman v1: Post-Mortem & Retrospective

## 1. Introduction
Varman began as an ambitious attempt to build an adversarial shield against facial recognition and generative AI models. Originally inspired by the structure of AMOR (Adversarial Masking and Obfuscation Routine) and MIT's PhotoGuard, the goal was to provide an end-to-end pipeline that could protect user images prior to uploading them to social media platforms like Instagram.

## 2. The Trilemma
The core technical challenge we faced can be best described as the **Varman Trilemma**. In adversarial machine learning applied to real-world social media, you are forced to balance three opposing forces:

1.  **Perturbation Strength:** The noise must be mathematically significant enough to completely scramble the identity embeddings (e.g., FaceNet, ArcFace) or semantic embeddings (CLIP).
2.  **Photo Quality (Invisibility):** The noise must be bounded by a strictly small $L_\infty$ norm (`epsilon`) so the human eye cannot perceive it, ensuring the image remains aesthetically pleasing.
3.  **Uploadability (Compression Survival):** Social media platforms aggressively compress images (JPEG, WebP). This high-frequency compression destroys delicate adversarial noise.

### Where we were too optimistic
We initially believed we could beat the Trilemma by employing **Expectation over Transformation (EoT)**—specifically using `DiffJPEG` inside the Projected Gradient Descent (PGD) loop to simulate compression during the optimization phase. 

**The Reality:** Forcing the optimizer to build noise that survives extreme compression meant it had to use much larger, lower-frequency patterns. Even when we restricted `epsilon`, the noise became highly visible (manifesting as grainy, noisy artifacts or blurriness). We realized that *truly invisible* noise is mathematically fragile; making it robust inherently makes it visible.

## 3. Hardware Challenges & Memory Bottlenecks
We were constrained to a 4GB VRAM environment (RTX 2050).
- **The AMOR approach:** Attempting to backpropagate through an entire Stable Diffusion VAE (as done in PhotoGuard) instantly caused Out-of-Memory (OOM) errors.
- **The Fix:** We pivoted to a Dual-Surrogate Face Recognition Ensemble (FaceNet + ArcFace). This was highly memory-efficient, taking only ~250MB of VRAM, but introduced a new problem: we were scaling down the entire 4K canvas to 512x512, adding noise, and scaling it back up, which destroyed the image's sharpness.

## 4. The Turning Point: Pivoting to Semantic Disruption
Faced with the reality that "Instagram-proof" protection was fundamentally degrading image quality, we asked a hard question: *What is the actual goal?*

If the goal is to break modern Multimodal LLMs (like Gemini or GPT-4V), we don't need uploadability. We can supply the protected image directly via lossless PNG. 

By dropping the compression survival requirement:
1.  We dropped our `epsilon` budget to a microscopic `4.1/255`.
2.  We achieved an SSIM of `> 0.98` and an LPIPS of `0.03`, meaning the perturbation is visually imperceptible.
3.  We switched from a face-recognition surrogate to an OpenCLIP ViT-B/32 surrogate. This allowed us to move the embedding into a substantially different region of the surrogate semantic space (achieving a negative cosine similarity of `-0.13`).

## 5. What We Learned
- **Robustness vs. Invisibility:** We were unable to achieve both under our constraints. If it survives aggressive compression, you can probably see it. If you can't see it, compression will likely destroy it.
- **Surrogate Alignment:** Attacking FaceNet does not guarantee you will break Gemini. You must attack the surrogate closest to the target's architecture (ViTs and CLIP).
- **Simplicity Wins:** By removing MTCNN face masking, DiffJPEG, and complex scaling logic, the pipeline became infinitely faster, more stable, and mathematically provable.

## 6. What Remains Unproven
Transferability to proprietary MLLMs remains an open question. 

The current benchmarks demonstrate substantial disruption of OpenCLIP embeddings while preserving image quality. Whether these perturbations consistently transfer to black-box models like Gemini, GPT-4V, Claude, or future multimodal systems requires further empirical testing.

## 7. Legacy of AMOR
AMOR was the theoretical foundation, but Varman is the grounded, empirical reality. We stripped away the heavy dependencies (`mediapipe`, `insightface`, `blind-watermark`) to create a lightweight semantic embedding disruption engine. We failed to conquer the Trilemma, but by narrowing our scope and following the evidence, we built a highly focused and testable adversarial framework.
