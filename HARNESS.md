# Native AR essay — prose harness

Read this file before touching `src/how_far_native_ar_video.md` or `src/how_far_native_ar_video.zh.md`. Do not edit the essay until you have walked the loop below. Style references (read, do not copy layout): [TML, On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/).

## What this note is

A bilingual research note. Chinese must read as written Chinese. English must read as written English. Neither is a translation of the other; they keep the same claims.

Author voice: Weiyang Jin (HKU). Use **我们 / we**. Do not switch to 我 / I.

## Do not change

- The argument (native = pretrain \(p(v_t\mid v_{<t},c_{\le t})\) + next step is one network evaluation).
- Equations, citation numbers, BibTeX, figure injection in `build_site.py` unless the user asked.
- Flash is one model we trained in the current stack, not the definition of native AR. Judgments only. No leaked numbers, case names, SSIM, window sizes, or checkpoint IDs.
- Do not invent a specific lookahead like \(v_{t+12}\). Future frames are \(v_{>t}\) or \(v_{t+1}\).

## Voice

Write like a person thinking on the page: define the word, then use it, then go on. Complete sentences. Contrast once, then move. Uneven sentence length is fine.

Do not write:

- lockstep definitions: 「只有 A，而且 B，我们才叫它 C」
- antithesis closers: 「该学的不是模块，而是道路」, 「说是「X」，做的还是 Y」. Write the name and the procedure in one order: 现在叫「X」的做法仍是…
- leftover English calques: 「一次前向写出一个因子」, 「人还在回路里」, 「人还在边看边操作」, 「回传过梯度」, 「纸上写的每一步」 as a refrain. 「纸上写的 / as written」is “生成下一步也不止需要一次前向”. 「看见谁」is *who is visible* — 谁 is for people; write 测试时能看见哪些帧. 「生成一次前向次数是」is *forwards per generation*; write 整段生成要跑 \(N\times K\) 次前向. 「并没有变成 \(f_\theta\)」treats a formula as a destination, same class as 到不了 \(p\)；write 并没有变成一次前向的 \(f_\theta(v_{<t})\). 「左边 / 右边」naming an equation is *left-hand side*; name the thing (时间上的乘积 / 训练这边是数据). 「地位不一样」is *same status*；write 不是一类东西. 「能力花在」is *spend capacity*；write 把容量用在. 「吐出」is *emit*；write 写出. 「传回去」is *gradient flows back*；write 这段历史要接得上梯度. 「现成的特征」is *free features*；write 把未来帧也当特征用. 「现在要测的变成了」is *the test has become*；write 现在面对的已经是. 「参数化的仍是」is *still parameterizes*；write 网络拟合的仍是. 流水线里的「强迫」is *force*；write 因果. 「把它落到 DiT 上」is *land that factorization on*；write 在 DiT 上再做这件事. 「绑在向量场上」is *tied to*；write 还在算 \(K\) 步向量场. 「往下跑」is *long-horizon runs*；write 长程生成时. 「从第一秒开始量，一直量到」is *Measure from … past*；write 从第一秒量到. 「扫一轮」is *another sweep*；write 再搜一轮. 「走不走得通」is *whether the path works*；write 这条路通不通. 「一条已经双向的 DiT」is *an already bidirectional DiT* as a countable line；drop 一条. 「接收跨块一致性的信号」is *receive a signal*；write 用跨块一致性来监督. 「操作过程中会进来」is *arrives as you go*；write 中途会碰到. 「缺口露出来一层、补一层」is *patched in the order gaps show up*；write 缺一层补一层. 「生成的就不是 \(B_i\)，而是一段常微分方程」and 「生成一步仍是一段微分方程」treat a step as *being* an ODE；write 写出来的并不是直接的 \(B_i\)，而是积一段方程 / 写出下一步仍要积一段微分方程.
- unexplained numbers, punchy metaphors (借口 / 戴上 mask / 赢的就是这一块 / 站在门口)
- 本文 / this paper. This is a note. 这里 / in this note / we.
- a conclusion that retells the whole essay. Restate the two conditions, that later patches do not reach them, and what has to change at pretraining. Do not make unpublished Flash runs the scope of “native.”
- telegram closers: a colon after 说得通 / 仍然是 that dumps the rest of the argument; stacked punches like 「现在要测的是流。」「事后再补也到不了。」「块内损失让同一块更好看。」. If a sentence is only a label plus a formula, write the reason in the same breath.
- broken Chinese continuity: related claims stay in one sentence, joined by commas / 而 / 所以. Do not copy English period-stacking. Split only when the thought turns. 「原生自回归是两件事，预训练拟合 p，写出下一步只需一次前向」is one breath; 「是两件事。预训练拟合 p。写出下一步，网络只跑一次。」is not. 「写出下一步时网络只跑一次」is still a calque of *when writing the next step the network runs once*; write 写出下一步只需一次前向.
- English continuity needs explicit subjects and causal links. Do not jump from “Language models…” to “Video left…” to the abstract subject “Catching up,” or use distant pointers such as “imitated it” and “those two conditions.” Name the actor and the mechanism: video generation moved away from next-frame prediction for short-clip quality; later patches make bidirectional clip models stream but do not change the condition fitted in pretraining.
- In English, prefer a concrete verb over a compressed noun chain: “the network predicts the vector field” rather than “the network parameterizes \(v_\theta\),” “sampling requires \(K\) steps” rather than “the model has integration length \(K\),” and “user actions arrive during generation” rather than “a person is still acting in the world.” Keep one grammatical subject through a causal sequence. Avoid anthropomorphic field-level subjects (“video left,” “the field asked”), dangling summaries (“what remains is,” “the goal now includes”), and unexplained demonstratives at paragraph boundaries.

TML is a **style** reference: calm we-voice, claim then why, one analogy at most. It is not a layout to copy (no hero, no sentence-tree cards).

## The loop (do it, do not skip)

1. **Read aloud once** in the language you edited. Mark any sentence you would not say to a colleague.
2. **Logic pass.** Every 所以 / 于是 / so / therefore must follow from the previous sentence. If the dash or colon hides two claims, split them. A colon after 说得通 / 仍然是 / 要的是 / 两件事 is almost always two claims. If 两者 / 这两条 has no nearby noun, the break is wrong.
3. **Leftover-AI pass.** Search for: 只有…才, 不是 A 而是 B, 因子, 借口, 回路, 门口, 本文, 第一组, 第二组, 实验里, 该赢, 否定原生, 还能做的，是, 口头上的, 要测的是流, 现在要测的, 变成流, 在那上面是对的, 不要…一遍, 说是「 , 做的还是, 在那之前, 纸上写的, 写下来的分解, 一条分解, 时间分解, 同一条分解, 时间乘积, 生成一步也不是, 看见谁, 前向次数, 并没有变成 \(f, 并没有变成 \(p, 到不了 \(p, 能力花在, 吐出, 传回去, 现成的特征, 人还在, 参数化的, → 强迫, 落到, 绑在, 往下跑, 开始量, 扫一轮, 地位不一样, 左边这个, 左边是数据, 走不走得通, 一条已经双向, 接收跨块, 操作过程中会进来, 缺口露出来一层, 生成的就不是, 生成一步仍是, 写出下一步网络只跑, 写出下一步时网络, 时网络只跑, 网络只跑一次, 学到的函数, 干不干净, 输给, 已经发生的过去, 这件事还没做, 先看训练, *as written*, *do not reach \(p*, *Neither reaches \(p*, *first experiment*, *second experiment*, *supposed to win*, *what remains is*, *the test now is*, *the test is a stream*, *Once the test is*, *collapses to*, *what to learn is not*, *misses both ends*, *the intuition is not hard*, *is not one network evaluation*, *who is visible*, *forwards per generation*, *left-hand side*, *write, for*, *lost out to*, *is still denoise*, *a person is still acting*, *Video left*, *Catching up*, *imitated it*, *those two conditions*, *The goal now includes*, *How Far We Are From*. A design sentence that ends in 不要 X is usually 不用 / 不必: the point is that the extra step is unnecessary, not a prohibition. 在那之前 needs a noun (没改之前 / 改预训练之前), not a hanging 那. Bare 流 for the test is 流式生成 (keep 流匹配). 到不了 \(p(\cdot)\) / 并没有变成 \(f_\theta\) treats a formula as a place; write 到不了拟合 \(p(\cdot)\) 的目标 / 并没有变成一次前向的 \(f_\theta(v_{<t})\). 分解 for temporal factorization is 「下一帧只看过去」. 看见谁 → 测试时能看见哪些帧.
4. **Align, do not translate.** After Chinese changes, fix the matching English claim — same thought, not the same syntax.
5. **Rebuild.** `python3 build_site.py` from the blog root. Check that KaTeX still has \(v_{>t}\) and \(w_{<i}\).
6. Stop when a second read-aloud finds nothing you would rewrite. If you still want to “polish tone”, you are done — do not take another stylistic lap.

## After edits

Rebuild HTML. Commit and push to `origin` on this repo only when the user is iterating on the live site or asks. Never touch `/pfs/yaoweili/`.
