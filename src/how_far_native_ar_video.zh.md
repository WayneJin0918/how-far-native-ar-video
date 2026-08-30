# 我们离原生自回归视频模型还有多远

语言模型里，自回归的意思其实早就清楚了。预训练拟合的就是

\[
p(w_{1:n})=\prod_i p(w_i\mid w_{<i}),
\]

每写下一个词，网络只跑一次。训练看见的、测试看见的，都是已经写出的词。没人会先训一个双向 BERT [6]，再蒸馏出一个「看起来像 GPT」的学生，然后说这就是原生自回归。推理时加一层因果 mask，也不算原生。

视频如果也用这个词，就写成：对 \(V=(v_1,\ldots,v_T)\)，

\[
p(V\mid c)=\prod_t p(v_t\mid v_{<t},c_{\le t}).
\]

\(c_t\) 是当时已经有的条件：文本、第一帧、相机位姿、按键。它可以进条件，但不等于可以去看还没发生的 \(v_{>t}\)。这里说的原生自回归有两层意思：预训练拟合的就是这个乘积；写出下一步，网络只跑一次。Echo-WM-Flash [[1]](https://arxiv.org/abs/2608.23189) 是我们按现在通行的扩散自回归训出来的。下面拿它当例子，看现在这套做法离这个定义还有多远。

---

## 图像没有时间；视频多出来的是 \(t\)

一张图里，左右是同时存在的。PixelRNN [4] 把似然写成光栅顺序 \(p(x)=\prod_{h,w}p(x_{hw}\mid x_{<hw})\)，只是给联合分布排个序。这个序是编码约定，不是时间。DiT [12] 后来把光栅顺序丢掉，在 \((h,w)\) 上做双向注意力，更接近一张照片。一帧里面空间双向，并不妨碍时间上的约束。

视频多出来的是 \(t\)。Video Pixel Networks [5] 在像素级就已经把 \(t\) 当序列，而不是把时长 \(T\) 拉成一张更高的图。写 \(v_t\) 的时候，\(v_{t+1}\) 还不存在：不是画面还没画完，是下一帧还没发生。所以同一个似然里，两种顺序的地位不一样：

\[
p(V)=\prod_t p\bigl(v_t\mid v_{<t}\bigr),\qquad
p(v_t\mid v_{<t})\;\text{在 }(h,w)\text{ 上可以双向。}
\]

左边这个乘积来自时间，后面的帧这一刻还不存在。右边帧内双向，只是当前帧怎么表示。把 \(T\) 当成更长的 \(H\)，就是把「还没发生」当成「画面另一边还没画完」。短短几秒的片子上，这个差别几乎看不出来：未来帧就在同一条训练样本里，用上它们画质更好。Sora [14]、Wan [16]、LTX [15] 靠的就是这个。它们拟合的是整段短片的联合分布，所以每个时刻的条件是

\[
q_{\mathrm{bi}}(v_t\mid V_{\setminus t},c).
\]

测试变成流以后，需要的条件就变了。接着往下写、\(c_t\) 随着操作到来、生成比训练时见过的窗口更长，这时要的是

\[
q_{\mathrm{AR}}(v_t\mid v_{<t},c_{\le t})=p_{\mathrm{data}}(v_t\mid v_{<t},c_{\le t}).
\]

训练时能看见哪些变量，学到的函数就不一样。\(c_t\) 换成文本、图像或位姿，只是条件里多一项，\(V_{\setminus t}\) 不会变成 \(v_{<t}\)。世界模型 [22]–[25] 很早就把这个错位露出来了，因为人还在里面操作。更长的视频、更低延迟的续写，碰到的还是这件事。

VideoGPT [7]、TATS [9] 已经在离散符号上写过 \(\prod_t p(v_t\mid v_{<t})\)。画质后来输给 MaskGIT [8]、MAGVIT [11]、Phenaki [10]，再输给按整段短片训的 DiT 级双向模型。图好看了，时间上的分解却丢掉了。产品变成流之后，领域没有回到「时间上只看过去」的预训练，而是在问：已经训好的 \(q_{\mathrm{bi}}\)，怎么改一改，让测试看起来像 \(q_{\mathrm{AR}}\)。后面的流水线，就是在做这件事。

---

## 训练看见了未来，生成一步也不是一次网络计算

按这个定义，现在通行的扩散自回归两条都不成立。训练时条件看见了未来。纸上写的每一步，也不是只跑一次网络。

先看训练。本该是 \(q_{\mathrm{train}}(v_t\mid\cdot)=p_{\mathrm{data}}(v_t\mid v_{<t})\)。如果拟合的是 \(q_{\mathrm{bi}}\)，推理时再加因果 mask（当前块不许看后面的块），只是测试时把训练用过的变量删掉。模型从来没有在「只有过去」这个条件下训练过。

再看生成。时间切成 \(B_1,\ldots,B_N\) 之后，写下来的分解是 \(p(B_{1:N})=\prod_i p_\theta(B_i\mid B_{<i})\)。可这一步如果做成

\[
x_{\sigma_K}\sim\mathcal{N}(0,I),\quad
x_{\sigma_k}=x_{\sigma_{k+1}}+(\sigma_k-\sigma_{k+1})\,v_\theta(x_{\sigma_{k+1}},\sigma_{k+1}\mid B_{<i},B_i^{\mathrm{noisy}}),
\]

生成的就不是 \(B_i\)，而是一段常微分方程。\(K\sim 30\) 是短片双向模型的积分长度；\(K=4\) 只是把同一种向量场的步数缩短，并没有变成 \(f_\theta(v_{<t})\)。

这两条缺口接着会变成一串后训练。训练已经看见未来，推理就只能加上因果 mask。训练时历史用的是干净数据（teacher forcing），测试时历史却是模型自己刚写出的块：

\[
q_{\mathrm{train}}(B_{<i})=p_{\mathrm{data}}(B_{<i}),\qquad
q_{\mathrm{infer}}(B_{<i})=p_\theta(B_{<i}).
\]

左边是数据，右边是模型自己刚写的。Bengio 等的 scheduled sampling [2]、Ranzato 等的序列级训练 [3]、Self Forcing [19]，处理的都是这个缺口。真要在「只有过去」下训，又贵又不稳。缺口还在，就再加一级：先让模型自己往下滚，在这段历史上做 DMD [28]；再把同样的损失加到已经带误差的更长历史上，\(\mathcal{L}_{\mathrm{long}}=\sum_k\mathcal{L}_{\mathrm{DMD}}^{(k)}\)。预训练从没在「只有过去」下训练过，这些阶段都是事后加上的。

生成一步仍是一段微分方程，所以要蒸馏。Diffusion Forcing [17]、CausVid [18]、Self Forcing [19]、Rolling Forcing [20]、MAGI-1 [21] 推理时都不看未来，块里面却还是用多步把 \(K\) 压下来。生成一次，前向次数是 \(N_{\mathrm{blocks}}\times K\)。到了 \(K=4\)，网络参数化的仍是 \(v_\theta(x,\sigma)\)。

训练条件没改，每一步仍是多步积分。生成又要比训练窗口更长，于是改的是推理时看得见的过去 \(\mathcal{H}_i\)：留几帧早期的当锚点、只留最近几块、窗口滑走时重置位置编码。StreamingLLM [13] 在语言里说过，滑动窗口需要锚点；FramePack [26]、WorldMem [27] 问的是 \(\mathcal{H}_i\) 里该留什么。这改变的是测试时看见谁，不是训练时拟合的条件。

现在通行的做法，是缺口露出来一层、补一层：

```
# 扩散自回归（现在通行）                 # 原生（定义要求的）
for i in 1..N:                          for t in 1..T:
    x ~ N(0, I)                             v_t = f_θ(v_<t, c_≤t)   # 一次前向
    for σ in [1.0, 0.75, 0.50, 0.25]:      cache.write(v_t)
        x = x + Δσ * v_θ(x, σ | B_<i, x)
    cache.write(x)
```

```
预训练    双向 DiT，整段短片              # 学 q_bi
→ 强迫    块因果 mask，干净历史           # 补「只看过去」；历史仍是数据
→ 短程    自己滚出来的历史 + DMD          # 补 K；历史仍短
→ 长程    拼接 + 固定窗口                 # 补长度
→ 搜索    窗口多长、要不要重置位置         # 补 H_i；仍不是模型参数
```

每一级都多一组要调的量。如果预训练从来没用过 \(v_{>t}\)，下一步又只跑一次网络，这些级就没有必要。

块该切多长，也不是可以随便选的。一块只有一帧，块内没有时间上的运动可积，多步积分只剩空间去噪。一块接近一秒，又变回带边界缓存的短片模型。三帧既能边写边出，又给块内双向去噪留了空间，所以成了常见折中。原生模型得在预训练就定下这个单元：一帧跑一次网络，或者一块跑一次网络，而不是对共享 mask 的三帧再积分一遍。

---

## 补充实验

我们在 Echo-WM-Flash [1] 上另外做了两组闭环、长程一致性实验，不在论文的公开表里。一组换后训练阶段，一组只改推理时看得见的过去 \(\mathcal{H}_i\)。

后训练如果优化的是块内去噪，同一块 \(B_i\) 里可以更好看，但不一定能改善块与块之间的条件。\(\prod_t p(v_t\mid v_{<t})\) 要的正是后者。只改 \(\mathcal{H}_i\)，改的是测试时看见谁，不是训练时拟合的条件。

第一组固定采样和窗口，比较不同后训练阶段的生成器。更晚的检查点，同一块 \(B_i\) 里纹理更干净，跨块的时间一致性并没有稳定跟着变好。我们做的这类后训练，优化的是块内目标。时间乘积要的，是块与块之间的条件。

第二组固定生成器，只改 \(\mathcal{H}_i\)：加长推理窗口，以及打开或关掉位置、位姿编码的重置。两者都不改训练条件。窗口和训练对不上，偏差出在块与块的接缝；重置和训练对不上，相对位置编码会指向另一套几何。跨块一致性没有因此稳定变好。改完之后，变的仍是测试时看见谁，不是把 \(q_{\mathrm{bi}}\) 换成 \(p(v_t\mid v_{<t})\)。

---

## 对原生设计意味着什么

Flash 上看到的，和前面的分析一致：块内后训练，以及推理时改窗口、开关重置，都到不了 \(p(v_t\mid v_{<t})\)。如果预训练就要拟合 \(p(v_t\mid v_{<t},c_{\le t})\)，下一步又只跑一次网络，后面几件事其实已经定了。

**块内后训练补不上跨块条件。** Flash 上更晚的后训练已经说明，块内损失会把能力花在 \(B_i\) 好不好看上。要把 \(\prod_t p(v_t\mid v_{<t})\) 学到，预训练的条件就得是过去。再加一个更好看的蒸馏阶段，补不上这个条件。继续预训练一条已经双向的 DiT，初始化可以留，但时间上的未来必须从第一步起就看不见。更干净的做法是同一套隐空间、帧内仍双向，时间 mask 从随机初始化就是因果的。VideoGPT [7] 写过这个分解，离散码本偏弱；缺的是把它落到 DiT 和现在的视频 VAE 上 [12][15][16]。损失可以还是流匹配，改的是条件：

\[
\mathcal{L}
=\mathbb{E}_{t,\sigma}\bigl\|
v_\theta(x_{\sigma,t},\sigma\mid v_{<t},c_{\le t})
-(\varepsilon-v_t)\bigr\|_2^2,
\]

\(\sigma\) 只加在当前单元上，梯度不经过 \(v_{>t}\)。少步一致性训练如果想当 \(K=1\) 的手段，得一开始就在 \(q_{\mathrm{AR}}\) 下训。先学 \(q_{\mathrm{bi}}\) 再压步数，能力又会花在块内积分上。第一组实验里，后训练擅长的正是这个，跨块一致性却没有跟着过来。自己滚一段再加 DMD，还可以继续补，但那是上一节写过的后训练，不是原生。

**加长窗口、开关重置，改的不是训练条件。** 第二组实验改了 \(\mathcal{H}_i\)，没改训练条件，跨块一致性也没有稳定变好。所以在原生模型里，看得见的过去应当写进模型结构：要么就是 \(v_{<t}\)（因果注意力的缓存），要么是参数里对 \(v_{<t}\) 的压缩，而且梯度能从这段历史传回去。固定「留多少锚点、留多少最近帧」，再去搜要不要重置位置编码，只在把 \(q_{\mathrm{bi}}\) 改造成 \(q_{\mathrm{AR}}\) 时才有必要。FramePack [26]、WorldMem [27] 问的是这段历史里该留什么。答案不该是再扫一轮和训练对不齐的窗口。

**时间单元在预训练定下来，测试时不再对块内积分。** 第一组实验改善的是块内去噪器的输出，说明现在这个学生的能力还绑在 \(K\) 步向量场上。如果说一块（或一帧）就是一步，测试时就不能再对这块做 \(K\) 步积分。一次网络计算写一帧，最接近语言里的一个词；一次网络计算写 \(L>1\) 帧，块内可以双向，每次吐出的片段更长。两种都可以，但 \(L\) 不能再当推理超参：改 \(L\) 就是改这一步有多长。块长如果还留给测试时折中，就会回到三帧加一段微分方程，后训练会再去优化这段方程。

**评价要看跨块，不能只看块内干不干净。** 第一组实验里，块内更干净和跨块更一致是两回事。如果用短片段、块内指标决定何时停训，还会再走到这条分叉上。应当用同一套权重、同一种只跑一次网络的生成方式，从第一秒量到超过训练窗口的长度；\(c_t\) 中途变了，再看条件是不是仍只依赖 \(v_{<t}\)。双向模型在十秒片段上继续更好看，本就是 \(q_{\mathrm{bi}}\) 该赢的地方，不能拿这个否定原生。

**数据也要按「过去已经发生」来切。** 等长的短片段把未来帧当成白送的特征。\(q_{\mathrm{AR}}\) 要求数据里的过去也是已经发生的过去：更长的连续镜头、带时间戳的 \(c_t\)、中途才出现的条件。不改训练条件，跨块一致性上不来，第二组实验已经见过。

还有几个问题没有答案，最好在再调窗口之前先想清楚。已经双向的大规模视频骨干，只做时间因果的继续预训练，要多久才能去掉 \(q_{\mathrm{bi}}\) 留下的习惯，要不要重新初始化？连续隐空间上 \(K=1\) 稳不稳，还是需要更强的离散或混合码本？原生目标下，一帧跑一次网络和三帧跑一次网络，哪一个更能同时保住运动和延迟？学出来的 \(\mathcal{H}_i\) 要不要直接接收跨块一致性的信号？第二组实验说明，没有这条信号的窗口搜索，并不改变模型学到的分解。

2021 年离开 VideoGPT 这条时间分解，当时说得通：要的是短片段，\(q_{\mathrm{bi}}\) 在那上面是对的。现在要测的是流。Flash [1] 上我们试过的后训练，以及推理时对 \(\mathcal{H}_i\) 的搜索，到不了 \(p(v_t\mid v_{<t})\)。还能做的，是预训练就改训练时能看见的变量。在那之前，口头上的「自回归视频」仍然是：先按整段短片去噪，再在推理时改成只看过去。

---

## 结论

原生自回归是两件事：预训练拟合 \(p(v_t\mid v_{<t},c_{\le t})\)，写出下一步只跑一次网络。现在通行的做法两条都不成立。Flash 上的两组实验核对过：块内后训练让同一块更好看，改窗口和重置只改测试时看见谁，跨块一致性都没有稳定跟着变好。

语言模型这几年走得快，是因为它一开始就按下一词来训，再把规模和数据堆上去。视频 2021 年离开了同一条分解，后来用因果 mask、少步积分和窗口搜索去模仿它。要跟上，就得在预训练改训练时能看见的变量。我们离这条路还很远。

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
