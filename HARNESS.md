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
- antithesis closers: 「该学的不是模块，而是道路」
- leftover English calques: 「一次前向写出一个因子」, 「人还在回路里」, 「回传过梯度」 as a refrain
- unexplained numbers, punchy metaphors (借口 / 戴上 mask / 赢的就是这一块 / 站在门口)
- 本文 / this paper. This is a note. 这里 / in this note / we.
- a conclusion that retells the whole essay. Restate the two conditions, that later patches do not reach them, and what has to change at pretraining. Do not make unpublished Flash runs the scope of “native.”

TML is a **style** reference: calm we-voice, claim then why, one analogy at most. It is not a layout to copy (no hero, no sentence-tree cards).

## The loop (do it, do not skip)

1. **Read aloud once** in the language you edited. Mark any sentence you would not say to a colleague.
2. **Logic pass.** Every 所以 / 于是 / so / therefore must follow from the previous sentence. If the dash or colon hides two claims, split them.
3. **Leftover-AI pass.** Search for: 只有…才, 不是 A 而是 B, 因子, 借口, 回路, 门口, 本文, 第一组, 第二组, 实验里, *first experiment*, *second experiment*, *collapses to*, *what to learn is not*, *misses both ends*, *the intuition is not hard*.
4. **Align, do not translate.** After Chinese changes, fix the matching English claim — same thought, not the same syntax.
5. **Rebuild.** `python3 build_site.py` from the blog root. Check that KaTeX still has \(v_{>t}\) and \(w_{<i}\).
6. Stop when a second read-aloud finds nothing you would rewrite. If you still want to “polish tone”, you are done — do not take another stylistic lap.

## After edits

Rebuild HTML. Commit and push to `origin` on this repo only when the user is iterating on the live site or asks. Never touch `/pfs/yaoweili/`.
