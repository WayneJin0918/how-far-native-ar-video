# How Far We Are From a Native Autoregressive Video Model

In language, the meaning of autoregression has been clear for a long time. Pretraining fits

\[
p(w_{1:n})=\prod_i p(w_i\mid w_{<i}),
\]

and each next word is one network evaluation. Training and test both see words already written. Nobody pretrains a bidirectional BERT [6], distills a student that “looks like GPT,” and calls that native. A causal mask at inference does not make it native either.

If video uses the same word, write, for \(V=(v_1,\ldots,v_T)\),

\[
p(V\mid c)=\prod_t p(v_t\mid v_{<t},c_{\le t}).
\]

\(c_t\) is a condition that is already there: text, a first frame, a camera pose, a key press. It can go into the condition. It does not let the model look at \(v_{>t}\), which has not happened yet. Here, native autoregression means two things: pretraining already fits this product, and writing the next step is one network evaluation. Echo-WM-Flash [[1]](https://arxiv.org/abs/2608.23189) is one model we trained in the diffusion-autoregressive stack that is now standard. The rest of this note is about that stack, and how far it is from this definition.

---

## An image has no time; video adds \(t\)

Left and right are present together in an image. PixelRNN [4] wrote \(p(x)=\prod_{h,w}p(x_{hw}\mid x_{<hw})\) only to give the joint some order. That order is a coding convention, not time. DiT [12] later dropped the raster order; bidirectional attention on \((h,w)\) is closer to a photograph. Keeping space bidirectional inside a frame is fine.

Video adds \(t\). Video Pixel Networks [5] already treated \(t\) as a sequence at the pixel level, rather than stretching duration \(T\) into a taller image. When \(v_t\) is written, \(v_{t+1}\) does not exist: not because the picture is unfinished, but because the next frame has not occurred. So inside one likelihood, the two orders do not have the same status:

\[
p(V)=\prod_t p\bigl(v_t\mid v_{<t}\bigr),\qquad
p(v_t\mid v_{<t})\;\text{may be bidirectional on }(h,w).
\]

The product on the left comes from time: later frames do not exist yet. Bidirectionality inside a frame is how the current frame is represented. Treating \(T\) as a longer \(H\) treats “has not happened” as “the other side of the picture is unfinished.” On a clip of a few seconds the difference is almost invisible. Future frames sit in the same training example, and using them improves quality. That is what Sora [14], Wan [16], and LTX [15] are good at. They fit the joint over a whole short clip, so the conditional at each time is

\[
q_{\mathrm{bi}}(v_t\mid V_{\setminus t},c).
\]

Once the test is a stream, the required condition changes. That includes continuation, a \(c_t\) that arrives as you go, and a horizon longer than the window seen in training. What is needed then is

\[
q_{\mathrm{AR}}(v_t\mid v_{<t},c_{\le t})=p_{\mathrm{data}}(v_t\mid v_{<t},c_{\le t}).
\]

Which variables training can see determines which function is learned. Swapping text, an image, or a pose for \(c_t\) adds one more term in the condition. It does not turn \(V_{\setminus t}\) into \(v_{<t}\). World models [22]–[25] made the mismatch visible early, because a person is still acting in the world. Longer video and low-latency continuation run into the same mismatch.

VideoGPT [7] and TATS [9] already wrote \(\prod_t p(v_t\mid v_{<t})\) over discrete symbols. Quality later lost out to MaskGIT [8], MAGVIT [11], Phenaki [10], then to bidirectional models trained on short clips at DiT scale. Pictures improved; the temporal factorization was dropped. When generation became a stream, the field did not return to pretraining that sees only the past in time. It asked how to reshape an existing \(q_{\mathrm{bi}}\) so the test looks like \(q_{\mathrm{AR}}\). The pipeline that followed is doing that work.

---

## Training sees the future, and the next step is not one forward

By that definition, the diffusion-autoregressive stack now in use satisfies neither condition. Training sees the future. Each step, as written, is not one network evaluation.

Start with training. It should be \(q_{\mathrm{train}}(v_t\mid\cdot)=p_{\mathrm{data}}(v_t\mid v_{<t})\). If the fit is \(q_{\mathrm{bi}}\), a causal mask at inference (the current block may not see later blocks) only removes at test time the variables training used. The model was never trained under “past only.”

Then generation. After cutting time into \(B_1,\ldots,B_N\), the claimed factorization is \(p(B_{1:N})=\prod_i p_\theta(B_i\mid B_{<i})\). If that step is implemented as

\[
x_{\sigma_K}\sim\mathcal{N}(0,I),\quad
x_{\sigma_k}=x_{\sigma_{k+1}}+(\sigma_k-\sigma_{k+1})\,v_\theta(x_{\sigma_{k+1}},\sigma_{k+1}\mid B_{<i},B_i^{\mathrm{noisy}}),
\]

what is generated is not \(B_i\), but a segment of an ordinary differential equation. \(K\sim 30\) is the integration length of the short-clip bidirectional model. \(K=4\) shortens the same vector field; it does not become \(f_\theta(v_{<t})\).

These two gaps then become a chain of post-training stages. Training already saw the future, so inference has to add a causal mask. Training history is clean data (teacher forcing); test history is the block the model just wrote:

\[
q_{\mathrm{train}}(B_{<i})=p_{\mathrm{data}}(B_{<i}),\qquad
q_{\mathrm{infer}}(B_{<i})=p_\theta(B_{<i}).
\]

The left-hand side is data. The right-hand side is what the model just wrote. Scheduled sampling [2], sequence-level training [3], and Self Forcing [19] all address this gap. Training under a true past-only condition is expensive and unstable. While the gap remains, another stage appears: first DMD [28] on a history the model rolled out itself, then the same loss on a longer history that already contains error, \(\mathcal{L}_{\mathrm{long}}=\sum_k\mathcal{L}_{\mathrm{DMD}}^{(k)}\). Pretraining was never run under “past only.” These stages were added afterwards.

Producing one step is still a differential equation, so people distill. Diffusion Forcing [17], CausVid [18], Self Forcing [19], Rolling Forcing [20], and MAGI-1 [21] all avoid the future at inference and still compress \(K\) with several steps inside the block. Forwards per generation scale as \(N_{\mathrm{blocks}}\times K\). At \(K=4\) the network still parameterizes \(v_\theta(x,\sigma)\).

The training condition is unchanged, and each step is still a multi-step integral. Generation then has to run past the window seen in training, so the visible past \(\mathcal{H}_i\) is edited: keep a few early frames as an anchor, keep only recent blocks, reset position encodings when the window moves. StreamingLLM [13] showed that a sliding window in language needs an anchor; FramePack [26] and WorldMem [27] asked what \(\mathcal{H}_i\) should hold. That changes who is visible at test time, not which conditional training fits.

The stack now in use is patched in the order the gaps show up:

```
# diffusion AR (current)                    # native (what the definition asks)
for i in 1..N:                              for t in 1..T:
    x ~ N(0, I)                                 v_t = f_θ(v_<t, c_≤t)   # one forward
    for σ in [1.0, 0.75, 0.50, 0.25]:          cache.write(v_t)
        x = x + Δσ * v_θ(x, σ | B_<i, x)
    cache.write(x)
```

```
pretrain  bidirectional DiT, full short clip  # learns q_bi
→ force   block-causal mask, clean history    # patch “past only”; history is still data
→ short   self-rollout + DMD                  # patch K; history still short
→ long    stitch + fixed window               # patch length
→ search  window size, whether to reset pos.  # patch H_i; still not parameters
```

Each extra stage adds quantities to tune. If pretraining never used \(v_{>t}\), and the next step is one network evaluation, those stages are unnecessary.

How long a block should be is not a free choice. If a block is one frame, there is no intra-block motion to integrate, and the leftover steps are only spatial denoising. If a block is near one second, the setup is again a short-clip model with a cache at the block boundary. Three frames are short enough to emit as you go, and still leave room for bidirectional denoising inside the block, which is why that length became common. A native model has to choose the unit at pretraining: one forward per frame, or one forward per block, rather than integrate again over three frames that share a mask.

---

## In our experience

We have trained models in this stack. Flash [1] is one of them. On closed-loop, long-horizon runs, two things kept coming back.

If post-training optimizes denoising inside a block, the same block \(B_i\) can look better without improving the condition between blocks. \(\prod_t p(v_t\mid v_{<t})\) is that between-block condition. Changing only \(\mathcal{H}_i\) changes who is visible at test time, not the condition fitted in training.

Later post-training, with sampling and the window held fixed, cleaned texture inside the same block \(B_i\). Consistency across blocks did not reliably follow. That post-training optimizes an intra-block objective. The temporal product needs a condition between blocks.

When we held the generator fixed and changed only \(\mathcal{H}_i\) — a longer window, positional and pose resets on or off — the training condition did not change. When the window disagrees with training, a discrepancy appears at the seam between blocks. When the reset disagrees, relative position encodings point at a different geometry. Consistency across blocks did not reliably improve. What changed is who is visible at test time, not a replacement of \(q_{\mathrm{bi}}\) by \(p(v_t\mid v_{<t})\).

---

## What that means for a native design

Intra-block post-training, and changing the window or the reset at inference, do not reach \(p(v_t\mid v_{<t})\). If pretraining is to fit \(p(v_t\mid v_{<t},c_{\le t})\) and the next step is one network evaluation, a few things are already fixed.

**Post-training inside a block does not supply the condition between blocks.** An intra-block loss spends capacity on how \(B_i\) looks. To learn \(\prod_t p(v_t\mid v_{<t})\), the pretraining condition has to be the past. Another distillation stage with cleaner texture does not supply that condition. Continued pretraining of an already bidirectional DiT can keep the initialization, but the future in time has to be invisible from the first step. A cleaner path keeps the same latent space and the same bidirectionality inside a frame, and makes the time mask causal from random initialization. VideoGPT [7] wrote this factorization; the discrete codebook was weak. What is missing is that factorization on a DiT and a current video VAE [12][15][16]. The loss can remain flow matching; the condition changes:

\[
\mathcal{L}
=\mathbb{E}_{t,\sigma}\bigl\|
v_\theta(x_{\sigma,t},\sigma\mid v_{<t},c_{\le t})
-(\varepsilon-v_t)\bigr\|_2^2,
\]

with \(\sigma\) applied only to the current unit and no gradient through \(v_{>t}\). Few-step consistency training can be a route to \(K=1\) only if it is trained under \(q_{\mathrm{AR}}\) from the start. Learning \(q_{\mathrm{bi}}\) and then compressing the step count spends capacity on integration inside the block again. That is what post-training is good at, and what does not transfer to consistency across blocks. Self-rollout plus DMD can keep patching this gap, but that is the post-training already described above, not a native model.

**A longer window, or turning a reset on and off, does not change the training condition.** Changing \(\mathcal{H}_i\) and not the training condition does not reliably improve consistency across blocks. So in a native model the visible past should be part of the model: either \(v_{<t}\) itself (the causal-attention cache), or a compression of \(v_{<t}\) inside the parameters, with a gradient that flows back through that history. A fixed “how many anchors, how many recent frames,” plus a search over whether to reset position encodings, is needed only when \(q_{\mathrm{bi}}\) is being reshaped into \(q_{\mathrm{AR}}\). FramePack [26] and WorldMem [27] asked what that history should contain. The answer cannot be another sweep of a window that disagrees with training.

**Choose the temporal unit at pretraining; do not integrate inside the block at test time.** An intra-block denoiser can look better while the student’s capacity is still tied to a \(K\)-step vector field. If one block (or one frame) is one step, test time cannot still integrate \(K\) denoising steps over that block. One forward per frame is the closest analogue of a word in language; one forward over \(L>1\) frames may stay bidirectional inside the block, and each emitted segment is longer. Both are allowed, but \(L\) cannot remain a test-time hyperparameter: changing \(L\) changes how long that step is. If block length is still a compromise at test time, the setup returns to three frames plus a leftover differential equation, and post-training will optimize that equation again.

**Measure across blocks, not by how clean a block looks.** Cleaner texture inside a block and better consistency across blocks come apart. If short clips and intra-block metrics decide when to stop training, that split will repeat. Use one set of weights and the same one-forward generation, measure from the first second past the training window, and when \(c_t\) changes mid-stream ask whether the condition still depends only on \(v_{<t}\). A bidirectional model that still looks better on a ten-second fragment is the interval where \(q_{\mathrm{bi}}\) is supposed to win. It does not show that the native path is wrong.

**Cut the data so that the past has already happened.** Fixed-length clips treat future frames as free features. \(q_{\mathrm{AR}}\) requires that the past in the data has already happened: longer continuous shots, timestamped \(c_t\), conditions that appear mid-sequence. Interventions that leave the training condition untouched do not bring consistency across blocks. We have seen this when only \(\mathcal{H}_i\) changes.

A few questions are still open, and they are worth asking before another window sweep. For a large bidirectional video backbone, how long does continued pretraining that is causal in time take to wash out the habits of \(q_{\mathrm{bi}}\), and is a re-initialization required? Is \(K=1\) stable in a continuous latent space, or is a stronger discrete or hybrid codebook needed? Under a native objective, which of one frame per forward and three frames per forward keeps both motion and latency? Should a learned \(\mathcal{H}_i\) receive a signal from consistency across blocks directly? Searching \(\mathcal{H}_i\) without that signal does not change the factorization the model learned.

Leaving the VideoGPT-style temporal factorization in 2021 made sense then: the goal was a short clip, and \(q_{\mathrm{bi}}\) is the right model there. The test now is a stream. Post-training after a bidirectional clip, and a search over \(\mathcal{H}_i\) at inference, do not reach \(p(v_t\mid v_{<t})\). What remains is to change, at pretraining, which variables training may see. Until then, what is called “autoregressive video” is still a model that denoises a whole short clip and is only later made to see the past.

---

## Conclusion

Native autoregression is two things: pretraining fits \(p(v_t\mid v_{<t},c_{\le t})\), and writing the next step is one network evaluation. The stack now in use satisfies neither. Patching afterwards does not get there. An intra-block loss cleans the same block. A longer window or a reset changes who is visible at test time. Neither supplies \(p(v_t\mid v_{<t})\).

Language models have moved quickly because they trained next-token prediction from the start, then scaled. Video left the same factorization in 2021, and later imitated it with a causal mask, a short integral, and a search over the window. Catching up means changing, at pretraining, which variables training may see. Another distillation stage will not do it. Until that change is made, we are still far from a native autoregressive video model.

---

## References

1. Zhang, S., Li, Y., Zhuang, J., Jin, W., et al. *EchoWM: Open and Enterable Omnimodal World Models.* arXiv:2608.23189, 2026. https://arxiv.org/abs/2608.23189
2. Bengio, S., Vinyals, O., Jaitly, N., and Shazeer, N. *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.* NeurIPS, 2015.
3. Ranzato, M., Chopra, S., Auli, M., and Zaremba, W. *Sequence Level Training with Recurrent Neural Networks.* ICLR, 2016. https://arxiv.org/abs/1511.06732
4. van den Oord, A., Kalchbrenner, N., and Kavukcuoglu, K. *Pixel Recurrent Neural Networks.* ICML, 2016. https://arxiv.org/abs/1601.06759
5. Kalchbrenner, N., van den Oord, A., Simonyan, K., et al. *Video Pixel Networks.* 2016. https://arxiv.org/abs/1610.00527
6. Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL, 2019.
7. Yan, W., Zhang, Y., Abbeel, P., and Srinivas, A. *VideoGPT: Video Generation using VQ-VAE and Transformers.* 2021. https://arxiv.org/abs/2104.10157
8. Chang, H., Zhang, H., Jiang, L., Liu, C., and Freeman, W. T. *MaskGIT: Masked Generative Image Transformer.* CVPR, 2022. https://arxiv.org/abs/2202.04200
9. Ge, S., Hayes, T., Yang, H., et al. *Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer.* ECCV, 2022. https://arxiv.org/abs/2204.03638
10. Villegas, R., Babaeizadeh, M., Kindermans, P.-J., et al. *Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions.* 2022. https://arxiv.org/abs/2210.02399
11. Yu, L., Cheng, Y., Sohn, K., et al. *MAGVIT: Masked Generative Video Transformer.* ICLR, 2023. https://arxiv.org/abs/2212.05199
12. Peebles, W., and Xie, S. *Scalable Diffusion Models with Transformers.* ICCV, 2023. https://arxiv.org/abs/2212.09748
13. Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. *Efficient Streaming Language Models with Attention Sinks.* ICLR, 2024. https://arxiv.org/abs/2309.17453
14. OpenAI. *Video generation models as world simulators (Sora).* 2024. https://openai.com/index/video-generation-models-as-world-simulators/
15. HaCohen, Y., Chiprut, N., Brazowski, B., et al. *LTX-Video: Realtime Video Latent Diffusion.* 2024. https://arxiv.org/abs/2501.00103  
    Lightricks. *LTX-2 / LTX-2.3.* 2025–2026.
16. Wan Team. *Wan: Open and Advanced Large-Scale Video Generative Models.* 2025. https://arxiv.org/abs/2503.20314
17. Chen, B., Martí Monsó, D., Du, Y., et al. *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion.* 2024. https://arxiv.org/abs/2407.01392
18. Yin, T., Zhang, Q., Zhang, R., Freeman, W. T., Durand, F., Shechtman, E., and Huang, X. *From Slow Bidirectional to Fast Autoregressive Video Diffusion Models (CausVid).* 2024. https://arxiv.org/abs/2412.07772
19. Huang, X., Li, Z., He, G., Zhou, M., and Shechtman, E. *Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion.* NeurIPS, 2025. https://arxiv.org/abs/2506.08009
20. Liu, K., Hu, W., Xu, J., Shan, Y., and Lu, S. *Rolling Forcing: Autoregressive Long Video Diffusion in Real Time.* 2025. https://arxiv.org/abs/2509.25161
21. Teng, H., Jia, H., Sun, L., et al. *MAGI-1: Autoregressive Video Generation at Scale.* Sand AI, 2025. https://arxiv.org/abs/2505.13211
22. Bruce, J., Dennis, M., Edwards, A., et al. *Genie: Generative Interactive Environments.* ICML, 2024. https://arxiv.org/abs/2402.15391
23. NVIDIA. *Cosmos World Foundation Model Platform for Physical AI.* 2025. https://arxiv.org/abs/2501.03575
24. Gao, Z., Wang, Q., Zhu, J., et al. *Infinite Worlds with Versatile Interactions (LingBot-World 2.0).* 2026. https://arxiv.org/abs/2607.07534
25. Zhu, H., Liu, H., Zhao, Y., et al. *SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer.* 2026. https://arxiv.org/abs/2605.15178
26. Zhang, L. *Packing Input Frame Context in Next-Frame Prediction Models (FramePack).* 2025. https://github.com/lllyasviel/FramePack
27. Xiao, Z., Lan, Y., Zhou, Y., Ouyang, W., Yang, S., Zeng, Y., and Pan, X. *WorldMem: Long-term Consistent World Simulation with Memory.* NeurIPS, 2025. https://arxiv.org/abs/2504.12369
28. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. *One-step Diffusion with Distribution Matching Distillation.* CVPR, 2024. https://arxiv.org/abs/2311.18828
