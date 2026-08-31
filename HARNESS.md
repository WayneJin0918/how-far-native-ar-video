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
- leftover English calques: 「一次前向写出一个因子」, 「人还在回路里」, 「回传过梯度」 as a refrain
- unexplained numbers, punchy metaphors (借口 / 戴上 mask / 赢的就是这一块 / 站在门口)
- 本文 / this paper. This is a note. 这里 / in this note / we.
- a conclusion that retells the whole essay. Restate the two conditions, that later patches do not reach them, and what has to change at pretraining. Do not make unpublished Flash runs the scope of “native.”
- telegram closers: a colon after 说得通 / 仍然是 that dumps the rest of the argument; stacked punches like 「现在要测的是流。」「事后再补也到不了。」「块内损失让同一块更好看。」. If a sentence is only a label plus a formula, write the reason in the same breath.
- broken Chinese continuity: related claims stay in one sentence, joined by commas / 而 / 所以. Do not copy English period-stacking. Split only when the thought turns. 「原生自回归是两件事，预训练拟合 p，写出下一步网络只跑一次」is one breath; 「是两件事。预训练拟合 p。写出下一步，网络只跑一次。」is not.

TML is a **style** reference: calm we-voice, claim then why, one analogy at most. It is not a layout to copy (no hero, no sentence-tree cards).

## The loop (do it, do not skip)

1. **Read aloud once** in the language you edited. Mark any sentence you would not say to a colleague.
2. **Logic pass.** Every 所以 / 于是 / so / therefore must follow from the previous sentence. If the dash or colon hides two claims, split them. A colon after 说得通 / 仍然是 / 要的是 / 两件事 is almost always two claims. If 两者 / 这两条 has no nearby noun, the break is wrong.
3. **Leftover-AI pass.** Search for: 只有…才, 不是 A 而是 B, 因子, 借口, 回路, 门口, 本文, 第一组, 第二组, 实验里, 该赢, 否定原生, 还能做的，是, 口头上的, 要测的是流, 变成流, 在那上面是对的, 不要…一遍, 说是「 , 做的还是, 在那之前, *first experiment*, *second experiment*, *supposed to win*, *what remains is*, *the test now is*, *the test is a stream*, *collapses to*, *what to learn is not*, *misses both ends*, *the intuition is not hard*. A design sentence that ends in 不要 X is usually 不用 / 不必: the point is that the extra step is unnecessary, not a prohibition. 在那之前 needs a noun (没改之前 / 改预训练之前), not a hanging 那. Bare 流 for the test is 流式生成 (keep 流匹配).
4. **Align, do not translate.** After Chinese changes, fix the matching English claim — same thought, not the same syntax.
5. **Rebuild.** `python3 build_site.py` from the blog root. Check that KaTeX still has \(v_{>t}\) and \(w_{<i}\).
6. Stop when a second read-aloud finds nothing you would rewrite. If you still want to “polish tone”, you are done — do not take another stylistic lap.

## After edits

Rebuild HTML. Commit and push to `origin` on this repo only when the user is iterating on the live site or asks. Never touch `/pfs/yaoweili/`.
