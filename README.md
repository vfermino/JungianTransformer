# JungianTransformer

A small, self-contained character-level decoder in C with a **Jungian mixer** —
a sequence-mixing layer whose *persona* routes each token over anonymous
archetype channels and whose *shadow* accumulates what the persona represses.
It is a testbed for an inference-time mechanism that lets an accumulated
"shadow" quietly redirect a frozen model across prompts.

Companion code for the paper ***Remembrance, not Memory: A Homeostatic Shadow
that Sequesters a Frozen Language Model*** (arXiv: _to appear_).

> Inspired by *I Built an LLM From Scratch* by CJ (Syntax,
> <https://www.youtube.com/watch?v=YmLp8qe87A0>). The premise of this project
> is that as the knowledge base for building models becomes widely accessible,
> measured small-scale experimentation is a realistic path to new ideas.

## Dependencies

Deliberately minimal — the whole model is one `.c` file:

- A C compiler (`clang` or `gcc`)
- **OpenMP** for multithreading (optional; the code compiles single-threaded
  without it). On macOS: `brew install libomp`.

No Python is required to run the model. `scripts/fetch_corpus.py` (standard
library only) rebuilds the corpus.

## Build

The architecture is selected entirely by compile-time flags.

**macOS (clang + libomp):**
```sh
L=$(brew --prefix libomp)
clang -O2 -Xpreprocessor -fopenmp -I$L/include -L$L/lib \
      -DLAYOUT_TYPE=LAYOUT_MAMBA_JUNGIAN -DJUNGIAN_V2=1 \
      -DSEQ_LEN=1024 -DBATCH=8 -DSTEPS=4000 \
      -o tf transformer.c -lm -lomp
```

**Linux (gcc):**
```sh
gcc -O2 -fopenmp \
    -DLAYOUT_TYPE=LAYOUT_MAMBA_JUNGIAN -DJUNGIAN_V2=1 \
    -DSEQ_LEN=1024 -DBATCH=8 -DSTEPS=4000 \
    -o tf transformer.c -lm
```

### Key flags

| flag | meaning |
|---|---|
| `-DLAYOUT_TYPE=` | mixer layout: `LAYOUT_MAMBA_JUNGIAN`, `LAYOUT_JUNGIAN`, `LAYOUT_MAMBA_CNN`, `LAYOUT_ATTN`, … |
| `-DJUNGIAN_V2=1` | persona with direct voice in the mixer output (`attended = p·v`) |
| `-DSTEER_AUG=1` | train the steering interface (self-supervised standing-bias augmentation) |
| `-DEMB_DIM=128` | embedding width (default 64); `FF_DIM` follows as `4·D` |
| `-DSEQ_LEN`, `-DBATCH`, `-DSTEPS` | context length, batch, training steps |
| `-DGRAD_CHECK=1` | build the finite-difference gradient checker instead of training |

## Corpus

The paper uses the five public-domain works of James Joyce on Project Gutenberg,
concatenated (2,500,916 chars, vocab 82). Rebuild it with:

```sh
python3 scripts/fetch_corpus.py        # writes corpus/joyce_all.txt
```

## Train

```sh
./tf corpus/joyce_all.txt --ckpt jj_v2_joyce.bin
```
Checkpoints are written every 50 steps and resumed automatically via `--ckpt`.

## Reproduce the experiments

Each probe is inference-only over a frozen checkpoint. Map to the paper:

| command | paper |
|---|---|
| `./tf corpus/joyce_all.txt --ckpt CKPT --emerge-test` | §5, archetypes emerge at inference (Table 1) |
| `./tf corpus/joyce_all.txt --ckpt CKPT --interface-test` | §7–§8, effector ceiling + learned interface (Tables 2–3) |
| `./tf corpus/joyce_all.txt --ckpt CKPT --repression-ce` | §9, sequestration via the CE detector (Table 4) |
| `./tf corpus/joyce_all.txt --ckpt CKPT --repression-test` | §6, the shadow across prompts (sampling detector) |

Gradient check (any backward-path change must pass):
```sh
clang -O2 -DLAYOUT_TYPE=LAYOUT_MAMBA_JUNGIAN -DJUNGIAN_V2=1 -DGRAD_CHECK=1 \
      -o gc transformer.c -lm && ./gc      # -> PASS
```

## The Jungian mixer, briefly

For hidden state `c_t` and learned archetype queries `A_k`:

```
sim_t   = <c_t, A_k> + β · shadow_{t-1}        (steering: the repressed feeds back)
p_t     = softmax(sim_t / τ)                   (persona: conscious routing)
shadow_t = α · shadow_{t-1} + (1-α)(1 - p_t)/(K-1)   (shadow: an EMA of the repressed)
```

The shadow is `O(K)` floats; persisted across prompts it becomes a per-session
memory that inclines — but, once trained, does not dominate — the next choice.

## Citation

```bibtex
@article{fermino2026remembrance,
  title  = {Remembrance, not Memory: A Homeostatic Shadow that Sequesters a Frozen Language Model},
  author = {Fermino, Victor},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

Copyright (C) 2026 Victor Fermino.

This program is free software: you can redistribute it and/or modify it under the
terms of the **GNU General Public License v3.0** as published by the Free Software
Foundation. See [`LICENSE`](LICENSE) for the full text.

It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.
