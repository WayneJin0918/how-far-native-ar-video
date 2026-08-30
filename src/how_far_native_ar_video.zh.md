# 我们离原生自回归视频模型还有多远

语言模型里，自回归的意思很清楚。预训练拟合的是

\[
p(w_{1:n})=\prod_i p(w_i\mid w_{<i}),
\]

每一个因子一次前向。训练和测试看见的都是已经写出的词。没有人会先训一个双向 BERT [6]，再蒸馏出一个「看起来像 GPT」的学生，然后把后者叫做原生自回归。推理时加上因果 mask，也不够。

视频上的对应写法是：对 \(V=(v_1,\ldots,v_T)\)，

\[
p(V\mid c)=\prod_t p(v_t\mid v_{<t},c_{\le t}).
\]

\(c_t\) 是已经发生的条件——文本、第一帧、相机位姿、按键——不是偷看尚未发生的 \(v_{t+12}\) 的许可。我们把预训练就拟合这个乘积、并且一次前向写出其中一个因子的模型，叫做原生自回归。Echo-WM-Flash [[1]](https://arxiv.org/abs/2608.23189) 是我们按现在通行的扩散自回归训出来的一条，后面用来看现行做法离这个定义有多远。

---

## 图像里没有必须遵守的时间；视频多出来的是 \(t\)

一张图没有必须遵守的时间。左右像素同时在场。PixelRNN [4] 把似然写成光栅顺序 \(p(x)=\prod_{h,w}p(x_{hw}\mid x_{<hw})\)，只为了让联合分布有一个顺序。这个顺序是编码约定。DiT [12] 把它拿掉，在 \((h,w)\) 上双向更接近一张照片。一帧之内空间保持双向，并不违反时间上的约束。

视频多出 \(t\)。Video Pixel Networks [5] 在像素级就已经把 \(t\) 当成序列，而不是把时长 \(T\) 拉成更长的高。生成 \(v_t\) 时，\(v_{t+1}\) 还不存在——不是「还没画上」，是尚未发生。所以同一个似然里，两种顺序的地位不同：

\[
p(V)=\prod_t p\bigl(v_t\mid v_{<t}\bigr),\qquad
p(v_t\mid v_{<t})\;\text{在 }(h,w)\text{ 上可以双向。}
\]

左边的乘积来自时间：后面的帧在生成这一帧时还不存在。右边帧内的双向是表示上的选择。把 \(T\) 当成更长的 \(H\)，等于把「尚未发生」当成「画面另一侧还没画完」。短短几秒的片段上，这个差别几乎看不出来：未来帧就在同一条训练样本里，用上它们画质更好。Sora [14]、Wan [16]、LTX [15] 赢的是这个局部。它们拟合的是整段短片的联合分布，因而每个时刻的条件是

\[
q_{\mathrm{bi}}(v_t\mid V_{\setminus t},c).
\]

测试一旦变成流——接着往下写、条件 \(c_t\) 现场到达、生成长度超过训练时见过的窗——需要的却是

\[
q_{\mathrm{AR}}(v_t\mid v_{<t},c_{\le t})=p_{\mathrm{data}}(v_t\mid v_{<t},c_{\le t}).
\]

训练时能看见的变量不同，学到的函数就不同。把 \(c_t\) 换成文本、图像或位姿，只改右边多出来的条件，并不会把左边的 \(V_{\setminus t}\) 变成 \(v_{<t}\)。世界模型 [22]–[25] 把这个错位暴露得很早，因为人还在回路里；更长的视频和低延迟续写，暴露的是同一件事。

接着走了一条本可以不走的路。VideoGPT [7]、TATS [9] 已经在离散符号上写过 \(\prod_t p(v_t\mid v_{<t})\)。画质输给 MaskGIT [8]、MAGVIT [11]、Phenaki [10]，再输给 DiT 规模、整段短片上训的双向模型。图更好看了，时间上的分解被丢掉。产品变成流之后，领域没有回到「时间上只看过去」的预训练，而是问：怎样把已经训好的 \(q_{\mathrm{bi}}\) 改造成测试时看起来像 \(q_{\mathrm{AR}}\)。后面的流水线是在回答这个问题。

---

## 训练看见了未来，一次前向也写不出一个因子

原生要求两件事同时成立：训练时条件只能看见已经发生的过去；每一个因子对应一次网络计算。现在通行的扩散自回归，两件都不成立。

**训练看见的不是只有过去。** 训练必须是 \(q_{\mathrm{train}}(v_t\mid\cdot)=p_{\mathrm{data}}(v_t\mid v_{<t})\)。若拟合的是 \(q_{\mathrm{bi}}\)，推理时加上因果 mask（当前块不许看后面的块）只是在测试时删掉训练用过的变量。模型从未在「只有过去」这个条件下出过梯度。

**一次前向写出的也不是一个因子。** 把时间切成块 \(B_1,\ldots,B_N\) 之后，声称的分解是 \(p(B_{1:N})=\prod_i p_\theta(B_i\mid B_{<i})\)。若这个因子被实现成

\[
x_{\sigma_K}\sim\mathcal{N}(0,I),\quad
x_{\sigma_k}=x_{\sigma_{k+1}}+(\sigma_k-\sigma_{k+1})\,v_\theta(x_{\sigma_{k+1}},\sigma_{k+1}\mid B_{<i},B_i^{\mathrm{noisy}}),
\]

真正的生成单元是一段常微分方程，不是 \(B_i\)。\(K\sim 30\) 是短片双向模型的积分长度；\(K=4\) 只是把同一种向量场的步数缩短，并没有变成 \(f_\theta(v_{<t})\)。

训练时已经看见未来，推理就只好装上 mask，训练再用干净的过去做 teacher forcing——「只有过去」贵，也不稳：

\[
q_{\mathrm{train}}(B_{<i})=p_{\mathrm{data}}(B_{<i}),\qquad
q_{\mathrm{infer}}(B_{<i})=p_\theta(B_{<i}).
\]

左边是数据，右边是模型自己刚写出的块。Bengio 等的 scheduled sampling [2]、Ranzato 等的序列级训练 [3]、Self Forcing [19]，说的都是这个缺口。缺口还在，就会再加一级：在模型自己滚出来的历史上做 DMD [28]，再把损失加到已经带误差的历史上，\(\mathcal{L}_{\mathrm{long}}=\sum_k\mathcal{L}_{\mathrm{DMD}}^{(k)}\)。预训练从没在「只有过去」下出过梯度，这些阶段是事后补上的。

一次前向仍是一段微分方程，就要蒸馏。Diffusion Forcing [17]、CausVid [18]、Self Forcing [19]、Rolling Forcing [20]、MAGI-1 [21] 都在推理时做到不看未来，块里面仍用多步把 \(K\) 压下来。一次生成的前向次数是 \(N_{\mathrm{blocks}}\times K\)。\(K=4\) 时，网络参数化的仍是 \(v_\theta(x,\sigma)\)。

训练条件没改、因子仍是多步积分，生成长度又要超过训练时见过的窗，就只能改推理时看得见的过去 \(\mathcal{H}_i\)：留住早期的几帧当锚点、只保留最近若干块、窗口滑走时重置位置编码。StreamingLLM [13] 在语言里指出，滑动窗口需要锚点；FramePack [26]、WorldMem [27] 问的是 \(\mathcal{H}_i\) 里该留什么。这些改的是测试时看见谁，不是训练时拟合的是哪一个条件。

现在通行的做法是按缺口出现的顺序一层层补上去的：

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

每一级多一组要调的量。若预训练从未使用 \(v_{>t}\)，并且在声称的时间单元上一次前向就写出一个因子，这些级就没有必要存在。

块该切多长，本身也是被逼出来的折中，不是可以随便选的设计。一块一帧，剩下的微分方程没有块内运动可积；一块接近一秒，又回到带边界缓存的短片模型。三帧既短到可以边写边吐，又给块内双向去噪留了空间，所以成了常见折中。原生模型必须在预训练就选定这个单元——一帧，或一块一次前向——而不是对共享 mask 的三帧再积分。

---

## 补充实验

我们在 Echo-WM-Flash [1] 上额外做了两组闭环、长程一致性实验，论文公开表里没有。一组改后训练阶段，一组只改推理时看得见的过去 \(\mathcal{H}_i\)。

后训练如果优化的是块内去噪，可以改善同一块 \(B_i\) 里的外观，而不必改善 \(\prod_t p(v_t\mid v_{<t})\) 所要求的、块与块之间的条件。只改 \(\mathcal{H}_i\)，改变的是测试时看见谁，不是训练时拟合的条件。

第一组固定采样和窗口，比较不同后训练阶段的生成器。更晚的检查点，在同一块 \(B_i\) 里纹理更干净；跨块的时间一致性没有稳定随之变好。我们做的这类后训练，优化的是块内目标。乘积分解要求的条件在块与块之间。

第二组固定生成器，只改变 \(\mathcal{H}_i\)：加长推理窗口，以及打开或关闭位置与位姿编码的重置。二者都不改变训练时拟合的条件。窗口和训练不一致时，偏差出现在块与块的接缝；重置和训练不一致时，相对位置编码对应另一套几何。跨块一致性没有因此稳定变好。改的是测试时看见谁，不是把 \(q_{\mathrm{bi}}\) 换成 \(p(v_t\mid v_{<t})\)。

---

## 对原生设计意味着什么

Flash 上的观察和前面的分析指向同一件事：块内后训练，以及推理期改窗口、开关重置，都到不了 \(p(v_t\mid v_{<t})\)。若预训练就要拟合 \(p(v_t\mid v_{<t},c_{\le t})\)，并在选定的时间单元上一次前向写出这个因子，下面几件事是跟着来的。

**块内后训练补不上跨块条件。** Flash 上更晚的后训练已经说明：块内损失会把能力花在 \(B_i\) 的外观上。要把 \(\prod_t p(v_t\mid v_{<t})\) 学到手，预训练的条件就必须是过去，而不是再加一个外观更好的蒸馏阶段。继续预训练一条已经双向的 DiT，可以保留初始化，但时间上的未来须从第一步起就看不见。更干净的做法是同一套隐空间、帧内仍双向，时间 mask 从随机初始化就因果。VideoGPT [7] 写过这个分解，离散码本偏弱；缺的是它落在 DiT 和现有视频 VAE 上 [12][15][16]。损失可以仍是流匹配，改的是条件：

\[
\mathcal{L}
=\mathbb{E}_{t,\sigma}\bigl\|
v_\theta(x_{\sigma,t},\sigma\mid v_{<t},c_{\le t})
-(\varepsilon-v_t)\bigr\|_2^2,
\]

\(\sigma\) 只加在当前单元上，梯度不经过 \(v_{>t}\)。少步一致性训练若要当作 \(K=1\) 的手段，须从一开始就在 \(q_{\mathrm{AR}}\) 下进行。先学 \(q_{\mathrm{bi}}\) 再压缩步数，会再次把能力花在块内积分上——这正是第一组实验里后训练擅长、却带不走跨块一致性的那件事。自身滚动加 DMD 可以继续补这个缺口，但那是上一节已经写过的补丁链，不是原生。

**加长窗口、开关重置，改的不是训练条件。** 第二组实验改变 \(\mathcal{H}_i\) 而不改变训练条件，跨块一致性没有稳定变好。因此原生里，看得见的过去必须是结构：要么就是 \(v_{<t}\)（因果注意力的缓存），要么是模型参数里对 \(v_{<t}\) 的压缩，并且梯度能通过这段历史传回去。固定「留多少锚点、留多少最近帧」、再搜索要不要重置位置编码，只在把 \(q_{\mathrm{bi}}\) 改造成 \(q_{\mathrm{AR}}\) 时才有必要。FramePack [26]、WorldMem [27] 问的是这段历史里该留什么；答案不能是再扫一轮和训练对不齐的窗口。

**时间单元在预训练选定，测试时不再对块内积分。** 第一组实验改善的是块内去噪器的输出，说明现在这个学生的能力仍绑在 \(K\) 步向量场上。原生若声称一块（或一帧）是一个因子，测试时就不能再对这块做 \(K\) 步积分。一次前向写一帧，最接近语言里的一个词；一次前向写 \(L>1\) 帧，块内可以双向，每次吐出的长度变粗。两者都可以，但 \(L\) 不能再当成推理超参数：改 \(L\) 就是改因子。块长若仍留给测试时折中，就会回到三帧加一段微分方程的局面，后训练会再次优化这段方程。

**评价要看跨块，不能只看块内是否干净。** 第一组实验里，块内更干净和跨块更一致是分开的。若用短片段、块内指标决定何时停训，会重复这条分叉。应当用同一套权重、同一种一次前向，从第一秒量到超过训练窗的长度，并在 \(c_t\) 中途改变时看条件是否仍只依赖 \(v_{<t}\)。双向模型在十秒片段上继续更好看，是 \(q_{\mathrm{bi}}\) 本来就该赢的区间，不能据此说原生这条路错了。

**数据也要按「过去已经发生」来切。** 等长的短片段把未来帧当成免费特征。\(q_{\mathrm{AR}}\) 要求数据里的过去也是已经发生的过去：更长的连续镜头、带时间戳的 \(c_t\)、中途才出现的条件。不改训练条件的干预带不来跨块一致性，第二组实验已经见过。

仍开放、而且值得在再调窗口之前回答的，是乘积本身上的选择。已经双向的大规模视频骨干，只做时间因果的继续预训练，要多久才能去掉 \(q_{\mathrm{bi}}\) 留下的习惯，是否必须重新初始化？连续隐空间上 \(K=1\) 是否稳定，还是需要更强的离散或混合码本？原生目标下，一帧一次前向和三帧一次前向，哪一个更能同时保住运动和延迟？学出来的 \(\mathcal{H}_i\) 要不要直接接收跨块一致性的信号——第二组实验表明，没有这条信号的窗口搜索，并不改变分解。

2021 年离开这条路，是因为当时要的是短片段，\(q_{\mathrm{bi}}\) 在那上面合理。现在要测的是流。Flash [1] 上我们试过的后训练，以及推理期对 \(\mathcal{H}_i\) 的搜索，到不了 \(p(v_t\mid v_{<t})\)。剩下可走的一步，是预训练就改训练时能看见的变量。在那之前，「自回归视频」仍是先按整段短片去噪、再在推理时改成只看过去的模型。

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
