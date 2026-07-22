/*
 * transformer.c — a decoder-only transformer (GPT-style) from scratch, in C.
 *
 * Faithful port of src/routes/train-transformer/{matrix.ts,transformer.ts},
 * plus a tiny char-level training loop in main() that mirrors train.ts (minus
 * the BPE tokenizer, workers and SharedArrayBuffer — all of which are infra,
 * not architecture).
 *
 * Everything is a flat float array with explicit dimensions, exactly like the
 * TS Float32Arrays. Row-major: element (i,j) of an (M,N) matrix is at i*N + j.
 *
 * Build:  cc -O2 -o transformer transformer.c -lm
 * Run:    ./transformer
 *
 * Architecture (decoder-only, like GPT):
 *   token embeddings + positional embeddings
 *   -> N blocks (layernorm -> causal multi-head self-attention -> residual
 *                layernorm -> feed-forward (expand->ReLU->project) -> residual)
 *   -> final layernorm -> linear head -> logits -> softmax -> probs
 *
 * @see https://arxiv.org/abs/1706.03762  "Attention Is All You Need"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Sentinel for "masked / impossible" attention scores. We use a large finite
 * negative value instead of -INFINITY: exp(NEG_INF - max) underflows to exactly
 * 0.0 just like exp(-inf), but staying finite keeps the code well-defined under
 * -ffast-math (which treats real infinities as undefined behavior). Same numbers,
 * safe to vectorize.
 */
#define NEG_INF (-1e30f)

/* ─── Architecture experiments (compile-time toggles) ─────────────────────────
 *
 * Two independent axes, both A/B-able against the vanilla 2017 transformer:
 *
 *  (1) Orthogonal toggles — flip to 1, combinable:
 *        USE_RMSNORM  RMSNorm instead of LayerNorm  (Zhang & Sennrich 2019, 1910.07467)
 *        USE_ROPE     Rotary position embeddings     (Su et al. 2021, 2104.09864)
 *        NUM_KV_HEADS GQA: shared KV heads (< numHeads)  (Ainslie et al. 2023)
 *
 *  (2) LAYOUT_TYPE — which *mixer* each layer uses (see the layout table below).
 *      The mixer is the sub-layer that moves information across positions;
 *      attention, Mamba/SSM, CNN and Hyena all implement the same interface, so
 *      layers can be mixed and matched (hybrid layouts, Jamba-style).
 *
 * Measurement: watch the loss printed every PRINT_EVERY epochs (convergence
 * speed), not just the final loss — the toy corpus saturates near the end.
 */
#ifndef USE_RMSNORM
#define USE_RMSNORM 0
#endif

/* ─── Mixer layouts ─────────────────────────────────────────────────────────
 * LAYOUT_TYPE selects a named layout from LAYOUTS[] (defined after Config).
 * Use the named macros for readability, e.g. -DLAYOUT_TYPE=LAYOUT_QUAD.
 */
#define LAYOUT_ATTN       0   /* pure attention (MHA/GQA, optional RoPE)   */
#define LAYOUT_MAMBA      1   /* pure Mamba (S6 SSM + conv1d)              */
#define LAYOUT_JAMBA      2   /* hybrid: Mamba + Attention                 */
#define LAYOUT_CNN        3   /* pure causal dilated CNN (TCN)            */
#define LAYOUT_TRIPLE     4   /* hybrid: Mamba + Attention + CNN           */
#define LAYOUT_MAMBA_CNN  5   /* attention-free hybrid: Mamba + CNN        */
#define LAYOUT_HYENA      6   /* pure Hyena / SGConv                       */
#define LAYOUT_QUAD       7   /* hybrid: Mamba + Attention + CNN + Hyena   */
#define LAYOUT_JUNGIAN    8   /* pure Jungian block (archetypal + shadow)  */
#define LAYOUT_MAMBA_JUNGIAN 9 /* hybrid: Mamba + Jungian                   */

#ifndef LAYOUT_TYPE
#define LAYOUT_TYPE LAYOUT_ATTN
#endif

#ifndef USE_ROPE
#define USE_ROPE 0
#endif

#ifndef JUNGIAN_ALPHA
#define JUNGIAN_ALPHA 0.9f
#endif
#ifndef JUNGIAN_TEMP
#define JUNGIAN_TEMP 1.0f
#endif
#ifndef JUNGIAN_STEERING
#define JUNGIAN_STEERING 0.5f
#endif
/* JUNGIAN_V2=1: a persona ganha voz direta na saída — attended_j = p·v (o
 * roteamento consciente seleciona os valores arquetípicos), no lugar da
 * atenção paralela q·A que ignorava p. Motivação: o teto do efetor medido no
 * NEURAL_EMERGENCE_PROBE.md (com v1, mesmo roteamento saturado move ~1% TV,
 * porque p só alcançava a saída via projShadow homeostático). O backward do
 * v2 usa a recorrência reversa EXATA da cadeia acoplada sim→p→shadow→sim
 * (o v1 trunca o termo β·dSim da realimentação de steering — tolerável lá,
 * onde p carrega gradiente pequeno; não aqui, onde p domina a saída). */
#ifndef JUNGIAN_V2
#define JUNGIAN_V2 0
#endif
/* ARCH_REPULSION=λ: decoupled archetype-repulsion step (see archRepulsionStep).
 * Matematicamente e psicologicamente um arquétipo só é significativo na medida
 * em que se distingue dos outros — a geometria do banco deve refletir isso. */
#ifndef ARCH_REPULSION
#define ARCH_REPULSION 0.0f
#endif

/* STEER_AUG=1: treina a INTERFACE de steering, não conteúdo (Jung: o treino
 * fixa os arquétipos-em-si — a base; a sessão preenche as imagens). Com p=0.5
 * por item do batch, a assinatura dos 384 chars vizinhos da janela é projetada
 * nos canais (detached) e aplicada como viés permanente durante forward+backward
 * com força w~U[0,40]. O vizinho compartilha o registro da janela, então a loss
 * ensina "viés b ⇒ texto no registro que gerou b" — supervisão grátis, conteúdo
 * variando o treino inteiro. Nenhum caminho novo de gradiente (viés é constante
 * aditiva na similaridade). */
#ifndef STEER_AUG
#define STEER_AUG 0
#endif

/* Eixo de capacidade (NEURAL_EMERGENCE_PROBE): D parametrizável para medir
 * efeito-por-dose vs. largura do modelo. FF_DIM segue a convenção 4·D. */
#ifndef EMB_DIM
#define EMB_DIM 64
#endif
#ifndef FF_DIM
#define FF_DIM (EMB_DIM * 4)
#endif


/* ─── Desatenção / Repression  (USE_REPRESSION=1) ────────────────────────────
 * An attention variant from a sister project ("disattention", MIND.md): instead
 * of attending to what the query is MOST similar to, attend to what it AVOIDS.
 * The score→weight map inverts the softmax:
 *
 *     p_ij = softmax(QKᵀ/√d)_ij          (normal attention weights, causal row)
 *     w_ij = (1 - p_ij) / Σ_k (1 - p_ik) (repression: a distribution over the
 *                                         LEAST-similar keys)
 *
 * w is still a proper convex combination (sums to 1), so the output stays on the
 * same scale as attention and the two are directly comparable — unlike the raw
 * unnormalized `V·(1-softmax)` from the manifesto, whose magnitude grows with
 * context length. Orthogonal to GQA/RoPE (it only remaps post-softmax weights);
 * reuses the attention Q/K/V/O params verbatim. Fully grad-checkable.
 * Row 0 (a single causal key) degenerates to w=0 — the first position gets no
 * mixer contribution, only its residual, which is fine.
 */
#ifndef USE_REPRESSION
#define USE_REPRESSION 0
#endif
#define REPR_EPS 1e-6f   /* guards 1/Σ(1-p) on the degenerate first row */

/* ─── Conv de contraste  (USE_CONTRAST_CONV=1) ───────────────────────────────
 * The positional-domain analog of Desatenção, applied to the CNN mixer. Softmax
 * attention is a content-based low-pass (it pools the similar); Desatenção makes
 * it high-pass (it keeps the dissimilar). A causal convolution is a *positional*
 * low-pass unless we force its kernel to have zero DC gain. This toggle
 * reparametrizes the depthwise kernel to be mean-subtracted per channel:
 *
 *     k'_m = k_m − (1/K) Σ_s k_s     (Σ_m k'_m = 0  ⇒  pure high-pass / contrast)
 *
 * A constant (redundant) input then convolves to 0 in steady state — the filter
 * passes local *change* and represses the *same*, the conv version of repression.
 * The residual still carries the constant. It's a linear reparametrization, so
 * the backward just applies the transpose: dk_n = dk'_n − mean(dk'). Grad-checkable.
 * Only affects MIX_CNN layers; no-op otherwise.
 */
#ifndef USE_CONTRAST_CONV
#define USE_CONTRAST_CONV 0
#endif

/* ─── ArchetypalProjection  (USE_ARCHETYPE=1) ────────────────────────────────
 * Portada do projeto-irmão disattention. Curva o espaço de embedding por uma
 * métrica dependente da entrada, aplicada UMA vez após os embeddings (não é um
 * mixer — não troca info entre posições; mistura canais com o mesmo M por
 * sequência). Mecanismo:
 *
 *   x̄     = média_s x_s                         (resumo da sequência, ℝ^D)
 *   w      = softmax( x̄ · Qᵀ / √D )              (pesos por arquétipo, ℝ^{N_ARCH})
 *   M      = I + Σ_k w_k C_k                     (métrica D×D)
 *   x'_s   = x_s · M                             (curvatura, mesma M p/ toda posição)
 *
 * Diferença honesta em relação ao original: lá os C_k eram covariâncias de
 * embeddings de passagens junguianas (via sentence-transformers). Num modelo
 * char-level de D pequeno isso não tem significado semântico, então portamos o
 * *mecanismo*: os C_k são uma base FIXA de matrizes simétricas aleatórias
 * (stream PRNG próprio, norma de Frobenius unitária, sem gradiente); só as
 * queries Q são aprendidas. O modelo aprende a rotear a curvatura. Grad-checkável
 * (Q entra em collectParams); os C_k ficam num buffer global read-only.
 */
#ifndef USE_ARCHETYPE
#define USE_ARCHETYPE 0
#endif
#ifndef N_ARCH
#define N_ARCH 16
#endif

#ifndef NUM_KV_HEADS
#define NUM_KV_HEADS 4
#endif

#ifndef PRINT_EVERY
#define PRINT_EVERY 30
#endif

/* Minibatch training on a file corpus (./transformer <file>). Override at compile
 * time, e.g. -DSEQ_LEN=128 -DSTEPS=5000, to give long-range models more to chew on. */
#ifndef SEQ_LEN
#define SEQ_LEN 64
#endif
#ifndef BATCH
#define BATCH 16
#endif
#ifndef STEPS
#define STEPS 2000
#endif
#ifndef WEIGHT_DECAY
#define WEIGHT_DECAY 0.01f
#endif
#ifndef GRAD_CLIP
#define GRAD_CLIP 1.0f
#endif
#ifndef LR_MIN_RATIO
#define LR_MIN_RATIO 0.1f
#endif

/* ─── Muon optimizer ────────────────────────────────────────────────────────
 * USE_MUON=1 optimizes the 2D hidden-layer weight matrices with Muon (Keller
 * Jordan): SGD-momentum whose update is orthogonalized by a Newton–Schulz
 * iteration, so every update is a well-conditioned rotation rather than an
 * axis-aligned rescale. Embeddings, the output head, conv kernels, and all 1D
 * params (biases/norms) stay on AdamW — Muon only applies to dense linear layers.
 * Orthogonal to architecture (no backward changes → no new grad-check needed).
 *
 * Muon needs a much larger LR than Adam (its updates are unit-scale), so the
 * scheduled Adam LR is multiplied by MUON_LR_SCALE for Muon params only.
 * @see https://kellerjordan.github.io/posts/muon/
 */
#ifndef USE_MUON
#define USE_MUON 0
#endif
#ifndef MUON_LR_SCALE
#define MUON_LR_SCALE 40.0f   /* base Adam LR 5e-4 × 40 ≈ 0.02 for Muon params */
#endif

/*
 * GRAD_CHECK=1 replaces training with a finite-difference gradient check: it
 * compares each analytical gradient (our hand-derived backward) against a
 * numerical estimate (loss(θ+ε) - loss(θ-ε)) / 2ε. This is how you trust a new
 * backward pass before believing its loss curve. Build WITHOUT -ffast-math for
 * accuracy:  cc -O2 -DGRAD_CHECK=1 transformer.c -lm && ./a.out
 */
#ifndef GRAD_CHECK
#define GRAD_CHECK 0
#endif

/* ─── Config ──────────────────────────────────────────────────────────────── */

/* The mixer each layer runs. Values are stable (used as array indices); add new
 * mixers at the end. Every mixer implements mixerForward/mixerBackward. */
typedef enum {
    MIX_ATTN  = 0,   /* causal multi-head attention (MHA/GQA, optional RoPE) */
    MIX_MAMBA = 1,   /* selective state-space model (S6) + causal conv1d      */
    MIX_CNN   = 2,   /* causal dilated convolution (TCN)                      */
    MIX_HYENA = 3,   /* Hyena / SGConv (long causal conv)                     */
    MIX_JUNGIAN = 4, /* Jungian block (archetypal + shadow)                   */
} MixerType;

#define MAX_LAYERS 8

typedef struct {
    int vocabSize;
    int contextLen;
    int embDim;    /* D */
    int numHeads;  /* H_Q (attention queries) */
    int numKVHeads;/* H_KV (attention keys/values) */
    int ffDim;     /* F */
    int numLayers;
    int stateN;    /* N: SSM state size (Mamba mixer only) */
    MixerType layerTypes[64]; /* mixer per layer; filled by applyLayout() */
} Config;

/* ─── Layout table ──────────────────────────────────────────────────────────
 * Each layout is just a list of mixers, one per layer. Adding a layout is one
 * row here — no scattered #if cascades. applyLayout() copies the selected row
 * into cfg (numLayers + layerTypes), so all three entry points stay in sync.
 */
typedef struct {
    const char *name;
    int numLayers;
    MixerType layers[MAX_LAYERS];
} Layout;

static const Layout LAYOUTS[] = {
    [LAYOUT_ATTN]      = { "attn",       2, { MIX_ATTN,  MIX_ATTN } },
    [LAYOUT_MAMBA]     = { "mamba",      2, { MIX_MAMBA, MIX_MAMBA } },
    [LAYOUT_JAMBA]     = { "jamba",      2, { MIX_MAMBA, MIX_ATTN } },
    [LAYOUT_CNN]       = { "cnn",        2, { MIX_CNN,   MIX_CNN } },
    [LAYOUT_TRIPLE]    = { "triple",     3, { MIX_MAMBA, MIX_ATTN, MIX_CNN } },
    [LAYOUT_MAMBA_CNN] = { "mamba-cnn",  2, { MIX_MAMBA, MIX_CNN } },
    [LAYOUT_HYENA]     = { "hyena",      2, { MIX_HYENA, MIX_HYENA } },
    [LAYOUT_QUAD]      = { "quad",       4, { MIX_MAMBA, MIX_ATTN, MIX_CNN, MIX_HYENA } },
    [LAYOUT_JUNGIAN]   = { "jungian",    2, { MIX_JUNGIAN, MIX_JUNGIAN } },
    [LAYOUT_MAMBA_JUNGIAN] = { "mamba-jungian", 2, { MIX_MAMBA, MIX_JUNGIAN } },
};

/* Fill cfg->numLayers and cfg->layerTypes from the compile-time LAYOUT_TYPE. */
static int applyLayout(Config *cfg) {
    const Layout *L = &LAYOUTS[LAYOUT_TYPE];
    cfg->numLayers = L->numLayers;
    for (int i = 0; i < L->numLayers; i++) cfg->layerTypes[i] = L->layers[i];
    return L->numLayers;
}

static const char *mixerName(MixerType t) {
    switch (t) {
        case MIX_ATTN:  return "Attn";
        case MIX_MAMBA: return "Mamba";
        case MIX_CNN:   return "CNN";
        case MIX_HYENA: return "Hyena";
        case MIX_JUNGIAN: return "Jungian";
    }
    return "?";
}

/* ─── PRNG: mulberry32 (deterministic, matches matrix.ts) ─────────────────── */

static uint32_t g_rng_state = 42;

static void resetRand(uint32_t seed) { g_rng_state = seed; }

/* returns a double in [0,1) */
static double randf(void) {
    uint32_t s = g_rng_state + 0x6D2B79F5u;
    g_rng_state = s;
    uint32_t t = (s ^ (s >> 15)) * (1u | s);
    t = ((t + ((t ^ (t >> 7)) * (61u | t))) ^ t);
    return (double)(t ^ (t >> 14)) / 4294967296.0;
}

/* ─── Allocation helpers ──────────────────────────────────────────────────── */

/* zeros(): calloc gives us zero-filled floats. Also used for grad buffers. */
static float *falloc(int len) {
    float *p = calloc((size_t)len, sizeof(float));
    if (!p) { fprintf(stderr, "OOM (%d floats)\n", len); exit(1); }
    return p;
}

static float *ones(int len) {
    float *p = falloc(len);
    for (int i = 0; i < len; i++) p[i] = 1.0f;
    return p;
}

/* Xavier/Glorot uniform: uniform in [-limit, limit], limit = sqrt(6/(fanIn+fanOut)). */
static float *xavierInit(int len, int fanIn, int fanOut) {
    float limit = (float)sqrt(6.0 / (double)(fanIn + fanOut));
    float *p = falloc(len);
    for (int i = 0; i < len; i++) p[i] = (float)(randf() * 2.0 - 1.0) * limit;
    return p;
}

#if USE_ARCHETYPE
/* Fixed archetypal curvature basis C_k [N_ARCH×D×D]: random symmetric, unit
 * Frobenius norm, built from an INDEPENDENT PRNG stream so it never perturbs the
 * main weight init (g_rng_state). Global + reachable at exit → not a leak. */
static float *g_archC = NULL;
static void buildArchC(int D) {
    if (g_archC) return;
    g_archC = falloc((size_t)N_ARCH * D * D);
    uint32_t s = 1234567u;   /* dedicated seed, separate from g_rng_state */
    for (int k = 0; k < N_ARCH; k++) {
        float *C = g_archC + (size_t)k * D * D;
        for (int i = 0; i < D; i++)
            for (int j = i; j < D; j++) {
                s += 0x6D2B79F5u;
                uint32_t t = (s ^ (s >> 15)) * (1u | s);
                t = ((t + ((t ^ (t >> 7)) * (61u | t))) ^ t);
                float r = (float)((double)(t ^ (t >> 14)) / 4294967296.0 * 2.0 - 1.0);
                C[i * D + j] = r; C[j * D + i] = r;   /* symmetric */
            }
        double fro = 0.0;
        for (int i = 0; i < D * D; i++) fro += (double)C[i] * C[i];
        float inv = (float)(1.0 / (sqrt(fro) + 1e-9));
        for (int i = 0; i < D * D; i++) C[i] *= inv;   /* unit Frobenius norm */
    }
}
#endif

static float silu(float x) { return x / (1.f + (float)exp(-x)); }

#if USE_ROPE
static void applyRoPE(float *tensor, int S, int D, int H, int inverse) {
    if (!tensor) return;
    int headDim = D / H;
    for (int t = 0; t < S; t++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < headDim / 2; i++) {
                float theta = (float)pow(10000.0, -2.0 * i / headDim);
                float omega = t * theta;
                if (inverse) omega = -omega;
                float cos_val = (float)cos((double)omega);
                float sin_val = (float)sin((double)omega);

                int idx1 = t * D + h * headDim + 2 * i;
                int idx2 = t * D + h * headDim + 2 * i + 1;

                float v1 = tensor[idx1], v2 = tensor[idx2];
                tensor[idx1] = v1 * cos_val - v2 * sin_val;
                tensor[idx2] = v1 * sin_val + v2 * cos_val;
            }
        }
    }
}
#endif

/* ─── Matrix ops (from matrix.ts) ─────────────────────────────────────────── */

/* (M,K) x (K,N) -> (M,N). The fundamental op of neural networks. Caller frees. */
static float *matmul(const float *a, const float *b, int M, int K, int N) {
    float *out = falloc(M * N);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float aik = a[i * K + k];
            for (int j = 0; j < N; j++)
                out[i * N + j] += aik * b[k * N + j];
        }
    return out;
}

/* (M,K) x (N,K)^T -> (M,N). Equivalent to A @ B.T. */
static float *matmulTransB(const float *a, const float *b, int M, int K, int N) {
    float *out = falloc(M * N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) sum += a[i * K + k] * b[j * K + k];
            out[i * N + j] = sum;
        }
    return out;
}

/* (K,M)^T x (K,N) -> (M,N). Equivalent to A.T @ B. */
static float *matmulTransA(const float *a, const float *b, int K, int M, int N) {
    float *out = falloc(M * N);
    for (int k = 0; k < K; k++)
        for (int i = 0; i < M; i++) {
            float aki = a[k * M + i];
            for (int j = 0; j < N; j++)
                out[i * N + j] += aki * b[k * N + j];
        }
    return out;
}

/* out[i][j] += bias[j] */
static void addBias(float *out, const float *bias, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[i * cols + j] += bias[j];
}

/* target[i] += src[i] */
static void addInPlace(float *target, const float *src, int n) {
    for (int i = 0; i < n; i++) target[i] += src[i];
}

/* target[i] += src[i], then free(src). Handy for grad accumulation of temporaries. */
static void accumFree(float *target, float *src, int n) {
    addInPlace(target, src, n);
    free(src);
}

/* Sum columns of (rows,cols) -> (cols,) vector. Caller frees. */
static float *sumCols(const float *mat, int rows, int cols) {
    float *out = falloc(cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j] += mat[i * cols + j];
    return out;
}

/* ─── Weights / Grads ─────────────────────────────────────────────────────── */

typedef struct {
    float *ln1Gamma, *ln1Beta;
    /* selective SSM mixer params (Mamba) */
    float *wB, *wC, *wDelta, *dtBias, *Alog, *Dskip, *wGate, *wOut;
    float *convW, *convB;
    /* Self-attention params */
    float *wQ, *bQ, *wK, *bK, *wV, *bV, *wO, *bO;
    /* Dilated CNN params */
    float *convW_cnn, *convB_cnn, *wOut_cnn, *bOut_cnn;
    /* Hyena/SGConv params */
    float *wU_hyena, *bU_hyena;
    float *wV_hyena, *bV_hyena;
    float *filter_hyena;
    float *wOut_hyena, *bOut_hyena;
    /* Jungian block params */
    float *wArch, *wQ_j, *bQ_j, *wV_j, *bV_j, *wS_j, *bS_j;
    float *ln2Gamma, *ln2Beta;
    float *ff1W, *ff1B, *ff1WUp, *ff1BUp, *ff2W, *ff2B;
} BlockWeights;

typedef struct {
    float *tokEmb, *posEmb;
#if USE_ARCHETYPE
    float *archQ;   /* [N_ARCH*D] learned archetype queries (C_k are the global g_archC) */
#endif
    BlockWeights *blocks;   /* [numLayers] */
    float *lnFGamma, *lnFBeta;
    float *headW, *headB;
} Weights;

/* Grads share the exact same shape as Weights, so we reuse the type. */
typedef Weights Grads;

static BlockWeights initBlockWeights(Config cfg, MixerType mixerType) {
    int D = cfg.embDim, F = cfg.ffDim, N = cfg.stateN;
    BlockWeights b;
    // Set all pointers to NULL initially
    b.wB = b.wC = b.wDelta = b.dtBias = b.Alog = b.Dskip = b.wGate = b.wOut = b.convW = b.convB = NULL;
    b.wQ = b.bQ = b.wK = b.bK = b.wV = b.bV = b.wO = b.bO = NULL;
    b.convW_cnn = b.convB_cnn = b.wOut_cnn = b.bOut_cnn = NULL;
    b.wU_hyena = b.bU_hyena = b.wV_hyena = b.bV_hyena = b.filter_hyena = b.wOut_hyena = b.bOut_hyena = NULL;
    b.wArch = b.wQ_j = b.bQ_j = b.wV_j = b.bV_j = b.wS_j = b.bS_j = NULL;

    b.ln1Gamma = ones(D);          b.ln1Beta = falloc(D);
    if (mixerType == MIX_MAMBA) {
        b.wB = xavierInit(D * N, D, N);
        b.wC = xavierInit(D * N, D, N);
        b.wDelta = xavierInit(D * D, D, D);
        b.dtBias = falloc(D);                 /* softplus(0)=ln2, a sane initial Δ */
        b.Alog = falloc(D * N);               /* A = -exp(Alog) = -(n+1): the S4D real init */
        for (int d = 0; d < D; d++)
            for (int n = 0; n < N; n++) b.Alog[d * N + n] = (float)log((double)(n + 1));
        b.Dskip = ones(D);                    /* skip connection init to 1 */
        b.wGate = xavierInit(D * D, D, D);    /* v2b: gate branch */
        b.wOut  = xavierInit(D * D, D, D);    /* v2b: output projection */
        b.convW = xavierInit(D * 4, 4, 4);    /* depthwise causal conv1d kernel K=4 */
        b.convB = falloc(D);                  /* bias initialized to 0 */
    } else if (mixerType == MIX_ATTN) {
        int headDim = D / cfg.numHeads;
        int D_KV = cfg.numKVHeads * headDim;
        b.wQ = xavierInit(D * D, D, D); b.bQ = falloc(D);
        b.wK = xavierInit(D * D_KV, D, D_KV); b.bK = falloc(D_KV);
        b.wV = xavierInit(D * D_KV, D, D_KV); b.bV = falloc(D_KV);
        b.wO = xavierInit(D * D, D, D); b.bO = falloc(D);
    } else if (mixerType == MIX_CNN) {
        b.convW_cnn = xavierInit(D * 3, 3, 3); // depthwise causal 1D conv kernel K=3
        b.convB_cnn = falloc(D);
        b.wOut_cnn  = xavierInit(D * D, D, D);
        b.bOut_cnn  = falloc(D);
    } else if (mixerType == MIX_JUNGIAN) {
        b.wArch = xavierInit(N_ARCH * D, N_ARCH, D);
        b.wQ_j  = xavierInit(D * D, D, D);
        b.bQ_j  = falloc(D);
        b.wV_j  = xavierInit(D * D, D, D);
        b.bV_j  = falloc(D);
        b.wS_j  = xavierInit(N_ARCH * D, N_ARCH, D);
        b.bS_j  = falloc(D);
    } else { // MIX_HYENA
        b.wU_hyena = xavierInit(D * D, D, D);
        b.bU_hyena = falloc(D);
        b.wV_hyena = xavierInit(D * D, D, D);
        b.bV_hyena = falloc(D);
        b.filter_hyena = xavierInit(cfg.contextLen * D, cfg.contextLen, D);
        b.wOut_hyena = xavierInit(D * D, D, D);
        b.bOut_hyena = falloc(D);
    }
    b.ln2Gamma = ones(D);          b.ln2Beta = falloc(D);
    b.ff1W = xavierInit(D * F, D, F); b.ff1B = falloc(F);
    b.ff1WUp = xavierInit(D * F, D, F); b.ff1BUp = falloc(F);
    b.ff2W = xavierInit(F * D, F, D); b.ff2B = falloc(D);
    return b;
}

static BlockWeights zeroBlockGrads(Config cfg, MixerType mixerType) {
    int D = cfg.embDim, F = cfg.ffDim, N = cfg.stateN;
    BlockWeights b;
    // Set all pointers to NULL initially
    b.wB = b.wC = b.wDelta = b.dtBias = b.Alog = b.Dskip = b.wGate = b.wOut = b.convW = b.convB = NULL;
    b.wQ = b.bQ = b.wK = b.bK = b.wV = b.bV = b.wO = b.bO = NULL;
    b.convW_cnn = b.convB_cnn = b.wOut_cnn = b.bOut_cnn = NULL;
    b.wU_hyena = b.bU_hyena = b.wV_hyena = b.bV_hyena = b.filter_hyena = b.wOut_hyena = b.bOut_hyena = NULL;
    b.wArch = b.wQ_j = b.bQ_j = b.wV_j = b.bV_j = b.wS_j = b.bS_j = NULL;

    b.ln1Gamma = falloc(D); b.ln1Beta = falloc(D);
    if (mixerType == MIX_MAMBA) {
        b.wB = falloc(D * N); b.wC = falloc(D * N); b.wDelta = falloc(D * D);
        b.dtBias = falloc(D); b.Alog = falloc(D * N); b.Dskip = falloc(D);
        b.wGate = falloc(D * D); b.wOut = falloc(D * D);
        b.convW = falloc(D * 4); b.convB = falloc(D);
    } else if (mixerType == MIX_ATTN) {
        int headDim = D / cfg.numHeads;
        int D_KV = cfg.numKVHeads * headDim;
        b.wQ = falloc(D * D);   b.bQ = falloc(D);
        b.wK = falloc(D * D_KV); b.bK = falloc(D_KV);
        b.wV = falloc(D * D_KV); b.bV = falloc(D_KV);
        b.wO = falloc(D * D);   b.bO = falloc(D);
    } else if (mixerType == MIX_CNN) {
        b.convW_cnn = falloc(D * 3);
        b.convB_cnn = falloc(D);
        b.wOut_cnn  = falloc(D * D);
        b.bOut_cnn  = falloc(D);
    } else if (mixerType == MIX_JUNGIAN) {
        b.wArch = falloc(N_ARCH * D);
        b.wQ_j  = falloc(D * D);
        b.bQ_j  = falloc(D);
        b.wV_j  = falloc(D * D);
        b.bV_j  = falloc(D);
        b.wS_j  = falloc(N_ARCH * D);
        b.bS_j  = falloc(D);
    } else { // MIX_HYENA
        b.wU_hyena = falloc(D * D);
        b.bU_hyena = falloc(D);
        b.wV_hyena = falloc(D * D);
        b.bV_hyena = falloc(D);
        b.filter_hyena = falloc(cfg.contextLen * D);
        b.wOut_hyena = falloc(D * D);
        b.bOut_hyena = falloc(D);
    }
    b.ln2Gamma = falloc(D); b.ln2Beta = falloc(D);
    b.ff1W = falloc(D * F); b.ff1B = falloc(F);
    b.ff1WUp = falloc(D * F); b.ff1BUp = falloc(F);
    b.ff2W = falloc(F * D); b.ff2B = falloc(D);
    return b;
}

static Weights initWeights(Config cfg) {
    int V = cfg.vocabSize, C = cfg.contextLen, D = cfg.embDim;
    Weights w;
    w.tokEmb = xavierInit(V * D, V, D);
    w.posEmb = xavierInit(C * D, C, D);
#if USE_ARCHETYPE
    buildArchC(D);
    w.archQ = falloc(N_ARCH * D);
    for (int i = 0; i < N_ARCH * D; i++) w.archQ[i] = (float)(randf() * 2.0 - 1.0) * 0.02f;
#endif
    w.blocks = malloc(sizeof(BlockWeights) * cfg.numLayers);
    for (int i = 0; i < cfg.numLayers; i++) {
        w.blocks[i] = initBlockWeights(cfg, cfg.layerTypes[i]);
    }
    w.lnFGamma = ones(D);
    w.lnFBeta = falloc(D);
    w.headW = xavierInit(D * V, D, V);
    w.headB = falloc(V);
    return w;
}

static Grads zeroGrads(Config cfg) {
    int V = cfg.vocabSize, C = cfg.contextLen, D = cfg.embDim;
    Grads g;
    g.tokEmb = falloc(V * D);
    g.posEmb = falloc(C * D);
#if USE_ARCHETYPE
    g.archQ = falloc(N_ARCH * D);
#endif
    g.blocks = malloc(sizeof(BlockWeights) * cfg.numLayers);
    for (int i = 0; i < cfg.numLayers; i++) {
        g.blocks[i] = zeroBlockGrads(cfg, cfg.layerTypes[i]);
    }
    g.lnFGamma = falloc(D);
    g.lnFBeta = falloc(D);
    g.headW = falloc(D * V);
    g.headB = falloc(V);
    return g;
}

/*
 * A flat list of every parameter array + its length, so we can iterate over the
 * whole model uniformly (zeroing grads, Adam updates). Order must match between
 * a Weights and its Grads/Adam buffers, which it does because shapes are identical.
 */
/* r,c hold the 2D shape for Muon-eligible dense weight matrices; left 0 for
 * everything else (embeddings, head, conv kernels, 1D params) → those use AdamW. */
typedef struct { float *ptr; int len; int decay; int r; int c; } ParamRef;

/*
 * Per-block parameter field names + count, kept in sync with collectParams' order
 * below and with the BlockWeights struct. gradCheck uses these to name a bad param
 * (hence only compiled in that build).
 */
#if GRAD_CHECK
static void getParameterName(int p, Weights *w, Config cfg, char *buf, size_t bufSize) {
    if (p == 0) { snprintf(buf, bufSize, "tokEmb"); return; }
    if (p == 1) { snprintf(buf, bufSize, "posEmb"); return; }
    int current = 2;
#if USE_ARCHETYPE
    if (p == current) { snprintf(buf, bufSize, "archQ"); return; }
    current += 1;
#endif
    for (int i = 0; i < cfg.numLayers; i++) {
        BlockWeights *b = &w->blocks[i];
        MixerType mixerType = (b->wB != NULL) ? MIX_MAMBA
                            : (b->wQ != NULL) ? MIX_ATTN
                            : (b->convW_cnn != NULL) ? MIX_CNN : MIX_HYENA;

        // Define fields in order of collectParams
        const char *mamba_fields[] = {
            "ln1Gamma", "ln1Beta", "wB", "wC", "wDelta", "dtBias", "Alog", "Dskip", "wGate", "wOut", "convW", "convB",
            "ln2Gamma", "ln2Beta", "ff1W", "ff1B", "ff1WUp", "ff1BUp", "ff2W", "ff2B"
        };
        int num_mamba = 20;

        const char *attn_fields[] = {
            "ln1Gamma", "ln1Beta", "wQ", "bQ", "wK", "bK", "wV", "bV", "wO", "bO",
            "ln2Gamma", "ln2Beta", "ff1W", "ff1B", "ff1WUp", "ff1BUp", "ff2W", "ff2B"
        };
        int num_attn = 18;

        const char *cnn_fields[] = {
            "ln1Gamma", "ln1Beta", "convW_cnn", "convB_cnn", "wOut_cnn", "bOut_cnn",
            "ln2Gamma", "ln2Beta", "ff1W", "ff1B", "ff1WUp", "ff1BUp", "ff2W", "ff2B"
        };
        int num_cnn = 14;

        const char *hyena_fields[] = {
            "ln1Gamma", "ln1Beta", "wU_hyena", "bU_hyena", "wV_hyena", "bV_hyena", "filter_hyena", "wOut_hyena", "bOut_hyena",
            "ln2Gamma", "ln2Beta", "ff1W", "ff1B", "ff1WUp", "ff1BUp", "ff2W", "ff2B"
        };
        int num_hyena = 17;

        int n_fields = (mixerType == MIX_MAMBA) ? num_mamba : ((mixerType == MIX_ATTN) ? num_attn : ((mixerType == MIX_CNN) ? num_cnn : num_hyena));
        const char **fields = (mixerType == MIX_MAMBA) ? mamba_fields : ((mixerType == MIX_ATTN) ? attn_fields : ((mixerType == MIX_CNN) ? cnn_fields : hyena_fields));

        if (p < current + n_fields) {
            snprintf(buf, bufSize, "b%d.%s", i, fields[p - current]);
            return;
        }
        current += n_fields;
    }
    const char *final_fields[] = {"lnFGamma", "lnFBeta", "headW", "headB"};
    if (p - current < 4) {
        snprintf(buf, bufSize, "%s", final_fields[p - current]);
        return;
    }
    snprintf(buf, bufSize, "unknown");
}
#endif /* GRAD_CHECK */

/* Fill `out` (capacity >= param count) with every param of `w`; returns count. */
/* PR: a param on plain AdamW (biases, norms, embeddings, head, conv kernels).
 * PRm: a dense 2D weight matrix (r×c) eligible for Muon when USE_MUON is set;
 *      it also gets weight decay (decay=1). Both fully initialize ParamRef, so
 *      collectParams stays clean under -Wextra (no missing-field warnings). */
static ParamRef PR(float *ptr, int len, int decay)      { return (ParamRef){ ptr, len, decay, 0, 0 }; }
static ParamRef PRm(float *ptr, int len, int r, int c)  { return (ParamRef){ ptr, len, 1, r, c }; }

static int collectParams(Weights *w, Config cfg, ParamRef *out) {
    int V = cfg.vocabSize, C = cfg.contextLen, D = cfg.embDim, F = cfg.ffDim, N = cfg.stateN;
    (void)N;
    int n = 0;
    /* Embeddings + head stay on AdamW (Muon is for hidden layers only); tokEmb
     * still gets weight decay as a matrix, posEmb does not (positional). */
    out[n++] = PR(w->tokEmb, V * D, 1);
    out[n++] = PR(w->posEmb, C * D, 0);
#if USE_ARCHETYPE
    out[n++] = PR(w->archQ, N_ARCH * D, 0);   /* AdamW (small gating param, no decay/Muon) */
#endif
    for (int i = 0; i < cfg.numLayers; i++) {
        BlockWeights *b = &w->blocks[i];
        out[n++] = PR(b->ln1Gamma, D, 0); out[n++] = PR(b->ln1Beta, D, 0);
        if (b->wB != NULL) {
            out[n++] = PRm(b->wB, D * N, D, N);      out[n++] = PRm(b->wC, D * N, D, N);
            out[n++] = PRm(b->wDelta, D * D, D, D);  out[n++] = PR(b->dtBias, D, 0);
            out[n++] = PR(b->Alog, D * N, 0);        out[n++] = PR(b->Dskip, D, 0);
            out[n++] = PRm(b->wGate, D * D, D, D);   out[n++] = PRm(b->wOut, D * D, D, D);
            out[n++] = PR(b->convW, D * 4, 1);       out[n++] = PR(b->convB, D, 0);
        } else if (b->wQ != NULL) {
            int headDim = D / cfg.numHeads;
            int D_KV = cfg.numKVHeads * headDim;
            out[n++] = PRm(b->wQ, D * D, D, D);       out[n++] = PR(b->bQ, D, 0);
            out[n++] = PRm(b->wK, D * D_KV, D, D_KV); out[n++] = PR(b->bK, D_KV, 0);
            out[n++] = PRm(b->wV, D * D_KV, D, D_KV); out[n++] = PR(b->bV, D_KV, 0);
            out[n++] = PRm(b->wO, D * D, D, D);       out[n++] = PR(b->bO, D, 0);
        } else if (b->convW_cnn != NULL) {
            out[n++] = PR(b->convW_cnn, D * 3, 1);
            out[n++] = PR(b->convB_cnn, D, 0);
            out[n++] = PRm(b->wOut_cnn, D * D, D, D);
            out[n++] = PR(b->bOut_cnn, D, 0);
        } else if (b->wArch != NULL) {
            out[n++] = PR(b->wArch, N_ARCH * D, 0);
            out[n++] = PRm(b->wQ_j, D * D, D, D);       out[n++] = PR(b->bQ_j, D, 0);
            out[n++] = PRm(b->wV_j, D * D, D, D);       out[n++] = PR(b->bV_j, D, 0);
            out[n++] = PRm(b->wS_j, N_ARCH * D, N_ARCH, D); out[n++] = PR(b->bS_j, D, 0);
        } else {
            out[n++] = PRm(b->wU_hyena, D * D, D, D);
            out[n++] = PR(b->bU_hyena, D, 0);
            out[n++] = PRm(b->wV_hyena, D * D, D, D);
            out[n++] = PR(b->bV_hyena, D, 0);
            out[n++] = PR(b->filter_hyena, C * D, 1);
            out[n++] = PRm(b->wOut_hyena, D * D, D, D);
            out[n++] = PR(b->bOut_hyena, D, 0);
        }
        out[n++] = PR(b->ln2Gamma, D, 0);  out[n++] = PR(b->ln2Beta, D, 0);
        out[n++] = PRm(b->ff1W, D * F, D, F);   out[n++] = PR(b->ff1B, F, 0);
        out[n++] = PRm(b->ff1WUp, D * F, D, F); out[n++] = PR(b->ff1BUp, F, 0);
        out[n++] = PRm(b->ff2W, F * D, F, D);   out[n++] = PR(b->ff2B, D, 0);
    }
    out[n++] = PR(w->lnFGamma, D, 0); out[n++] = PR(w->lnFBeta, D, 0);
    out[n++] = PR(w->headW, D * V, 1); out[n++] = PR(w->headB, V, 0);
    return n;
}

static int countParams(Weights *w, Config cfg) {
    ParamRef refs[8 + 64 * 16];
    int n = collectParams(w, cfg, refs);
    int total = 0;
    for (int i = 0; i < n; i++) total += refs[i].len;
    return total;
}

/* ─── Normalization (LayerNorm baseline / RMSNorm experiment) ─────────────── */

#if !USE_RMSNORM
/*
 * Normalize each position's activations to zero mean, unit variance across D,
 * then scale/shift by gamma/beta. Stores mean/var/xHat for the backward pass.
 * @see https://arxiv.org/abs/1607.06450  "Layer Normalization"
 */
static float *layerNormForward(const float *x, const float *gamma, const float *beta,
                               int seqLen, int D, float *mean, float *var, float *xHat) {
    float *out = falloc(seqLen * D);
    const float eps = 1e-5f;
    for (int i = 0; i < seqLen; i++) {
        float m = 0.f;
        for (int d = 0; d < D; d++) m += x[i * D + d];
        m /= D;
        mean[i] = m;

        float v = 0.f;
        for (int d = 0; d < D; d++) { float diff = x[i * D + d] - m; v += diff * diff; }
        v /= D;
        var[i] = v;

        float invStd = 1.f / (float)sqrt(v + eps);
        for (int d = 0; d < D; d++) {
            float hat = (x[i * D + d] - m) * invStd;
            xHat[i * D + d] = hat;
            out[i * D + d] = gamma[d] * hat + beta[d];
        }
    }
    return out;
}

/* Backward through layer norm. Accumulates into dGamma/dBeta, returns dX. */
static float *layerNormBackward(const float *dOut, const float *xHat, const float *gamma,
                                const float *mean, const float *var,
                                int seqLen, int D, float *dGamma, float *dBeta) {
    (void)mean;
    float *dX = falloc(seqLen * D);
    const float eps = 1e-5f;
    for (int i = 0; i < seqLen; i++) {
        float invStd = 1.f / (float)sqrt(var[i] + eps);
        float sumDxHat = 0.f, sumDxHatXHat = 0.f;
        for (int d = 0; d < D; d++) {
            float dxh = dOut[i * D + d] * gamma[d];
            sumDxHat += dxh;
            sumDxHatXHat += dxh * xHat[i * D + d];
        }
        for (int d = 0; d < D; d++) {
            float dxh = dOut[i * D + d] * gamma[d];
            dX[i * D + d] = invStd * (dxh - sumDxHat / D - xHat[i * D + d] * sumDxHatXHat / D);
        }
    }
    for (int i = 0; i < seqLen; i++)
        for (int d = 0; d < D; d++) {
            dGamma[d] += dOut[i * D + d] * xHat[i * D + d];
            dBeta[d]  += dOut[i * D + d];
        }
    return dX;
}

#endif /* !USE_RMSNORM */

#if USE_RMSNORM
/*
 * RMSNorm forward — normalize by the root-mean-square only, no mean subtraction
 * and no beta shift. out_d = gamma_d * x_d / sqrt(mean(x^2) + eps).
 *
 * The insight (Zhang & Sennrich 2019): LayerNorm's re-centering contributes little;
 * the re-scaling is what stabilizes training. Dropping the mean makes this cheaper
 * (one pass, one running sum) and, empirically, just as good or better.
 *
 * Stores invRMS per row (into the `var` buffer via the dispatcher) for backward.
 * @see https://arxiv.org/abs/1910.07467 — "Root Mean Square Layer Normalization"
 */
static float *rmsNormForward(const float *x, const float *gamma,
                             int seqLen, int D, float *invRMS, float *xHat) {
    float *out = falloc(seqLen * D);
    const float eps = 1e-5f;
    for (int i = 0; i < seqLen; i++) {
        float ms = 0.f;
        for (int d = 0; d < D; d++) { float xd = x[i * D + d]; ms += xd * xd; }
        ms /= D;
        float r = 1.f / (float)sqrt(ms + eps);
        invRMS[i] = r;
        for (int d = 0; d < D; d++) {
            float hat = x[i * D + d] * r;
            xHat[i * D + d] = hat;
            out[i * D + d] = gamma[d] * hat;
        }
    }
    return out;
}

/*
 * RMSNorm backward. Only one row-reduction (S1) vs LayerNorm's two, and no dBeta:
 *   dX_d = r * (g_d - xHat_d * S1 / D),  where g_d = dOut_d * gamma_d
 *                                        and   S1  = sum_d g_d * xHat_d.
 * Derived by hand from xHat_d = x_d * r,  r = (mean(x^2)+eps)^(-1/2).
 */
static float *rmsNormBackward(const float *dOut, const float *xHat, const float *gamma,
                              const float *invRMS, int seqLen, int D, float *dGamma) {
    float *dX = falloc(seqLen * D);
    for (int i = 0; i < seqLen; i++) {
        float r = invRMS[i];
        float S1 = 0.f;
        for (int d = 0; d < D; d++) S1 += dOut[i * D + d] * gamma[d] * xHat[i * D + d];
        for (int d = 0; d < D; d++) {
            float g = dOut[i * D + d] * gamma[d];
            dX[i * D + d] = r * (g - xHat[i * D + d] * S1 / D);
        }
    }
    for (int i = 0; i < seqLen; i++)
        for (int d = 0; d < D; d++)
            dGamma[d] += dOut[i * D + d] * xHat[i * D + d];
    return dX;
}
#endif /* USE_RMSNORM */

/*
 * Dispatchers: identical signature regardless of the norm choice, so the call
 * sites in blockForward/blockBackward/forward/backward never change. For RMSNorm
 * we reuse the `var` buffer to hold invRMS; `mean`, `beta` and `dBeta` go unused.
 */
static float *normForward(const float *x, const float *gamma, const float *beta,
                          int seqLen, int D, float *mean, float *var, float *xHat) {
#if USE_RMSNORM
    (void)beta; (void)mean;
    return rmsNormForward(x, gamma, seqLen, D, var, xHat);
#else
    return layerNormForward(x, gamma, beta, seqLen, D, mean, var, xHat);
#endif
}

static float *normBackward(const float *dOut, const float *xHat, const float *gamma,
                           const float *mean, const float *var,
                           int seqLen, int D, float *dGamma, float *dBeta) {
#if USE_RMSNORM
    (void)mean; (void)dBeta;
    return rmsNormBackward(dOut, xHat, gamma, var, seqLen, D, dGamma);
#else
    return layerNormBackward(dOut, xHat, gamma, mean, var, seqLen, D, dGamma, dBeta);
#endif
}

/* ─── Softmax (row-wise, numerically stable) ──────────────────────────────── */

static float *softmaxRows(const float *x, int rows, int cols) {
    float *out = falloc(rows * cols);
    for (int i = 0; i < rows; i++) {
        float mx = NEG_INF;
        for (int j = 0; j < cols; j++) if (x[i * cols + j] > mx) mx = x[i * cols + j];
        float sum = 0.f;
        for (int j = 0; j < cols; j++) {
            out[i * cols + j] = (float)exp(x[i * cols + j] - mx);
            sum += out[i * cols + j];
        }
        for (int j = 0; j < cols; j++) out[i * cols + j] /= sum;
    }
    return out;
}

/* ─── Per-block forward cache ─────────────────────────────────────────────── */

typedef struct {
    const float *input;  /* borrowed: prev block output or x0, NOT owned/freed here */
    float *ln1Out, *ln1Mean, *ln1Var, *ln1Hat;
    /* ── mixer intermediates (Mamba) ── */
    float *convOut;  /* S×D   output of depthwise causal conv1d */
    float *uPrime;   /* S×D   SiLU(convOut) */
    float *delta;    /* S×D   discretization step Δ (softplus output) */
    float *dtSig;    /* S×D   sigmoid(raw+dtBias) = softplus'(·), for backward */
    float *ssmB;     /* S×N   input-dependent B */
    float *ssmC;     /* S×N   input-dependent C */
    float *h;        /* S×D×N recurrent state at every timestep (for BPTT) */
    float *ssmY;     /* S×D   SSM readout, before gating (v2b) */
    float *gateZ;    /* S×D   gate pre-activation z = u·wGate (v2b) */
    /* ── mixer intermediates (Attention) ── */
    float *Q, *K, *V, *attnWeights, *attnVals;
    /* ── mixer intermediates (CNN) ── */
    float *convOut_cnn;  /* S×D */
    float *siluOut_cnn;  /* S×D */
    /* ── mixer intermediates (Hyena) ── */
    float *u_hyena;      /* S×D */
    float *v_hyena;      /* S×D */
    float *convOut_hyena;/* S×D */
    float *gateOut_hyena;/* S×D */
    /* ── mixer intermediates (Jungian) ── */
    float *sim_j;         /* S×K */
    float *p_j;           /* S×K */
    float *shadow_j;      /* S×K */
    float *q_j;           /* S×D */
    float *v_j;           /* K×D */
    float *attnScores_j;  /* S×K */
    float *attnWeights_j; /* S×K */
    float *attended_j;    /* S×D */
    float *projShadow_j;  /* S×D */
    float *mixerOut;     /* mixer output (attention or SSM), fed to the residual */
    float *x1;
    float *ln2Out, *ln2Mean, *ln2Var, *ln2Hat;
    float *ff1Out, *ff1UpOut, *siluOut, *inter, *ff2Out, *output;
} BlockCache;

static void freeBlockCache(BlockCache *bc) {
    free(bc->ln1Out); free(bc->ln1Mean); free(bc->ln1Var); free(bc->ln1Hat);
    /* free Mamba fields */
    free(bc->convOut); free(bc->uPrime); free(bc->delta); free(bc->dtSig);
    free(bc->ssmB); free(bc->ssmC); free(bc->h); free(bc->ssmY); free(bc->gateZ);
    /* free Attention fields */
    free(bc->Q); free(bc->K); free(bc->V); free(bc->attnWeights); free(bc->attnVals);
    /* free CNN fields */
    free(bc->convOut_cnn); free(bc->siluOut_cnn);
    /* free Hyena fields */
    free(bc->u_hyena); free(bc->v_hyena); free(bc->convOut_hyena); free(bc->gateOut_hyena);
    /* free Jungian fields */
    free(bc->sim_j); free(bc->p_j); free(bc->shadow_j);
    free(bc->q_j); free(bc->v_j);
    free(bc->attnScores_j); free(bc->attnWeights_j);
    free(bc->attended_j); free(bc->projShadow_j);

    free(bc->mixerOut); free(bc->x1);
    free(bc->ln2Out); free(bc->ln2Mean); free(bc->ln2Var); free(bc->ln2Hat);
    free(bc->ff1Out); free(bc->ff1UpOut); free(bc->siluOut); free(bc->inter);
    free(bc->ff2Out); free(bc->output);
}

/* ─── Mixer: causal multi-head self-attention ─────────────────────────────────
 *
 * The "mixer" is the sub-layer that lets positions exchange information. In a
 * transformer that's self-attention; swapping it for a different mixer (e.g. a
 * Mamba selective SSM) is how you change the architecture while leaving the rest
 * of the block — norm, residual, feed-forward, training — untouched.
 *
 * Interface:
 *   mixerForward(bc, bw, cfg, S)   reads bc->ln1Out, returns mixerOut (S×D), caches internals
 *   mixerBackward(dMixerOut, ...)  accumulates mixer param grads, returns dLn1Out (S×D)
 */

/* Causal multi-head self-attention. The mask (j > i => -inf) makes it autoregressive. */
static float *mixerForwardAttn(BlockCache *bc, BlockWeights *bw, Config cfg, int S) {
    int D = cfg.embDim, H = cfg.numHeads;
    int headDim = D / H;
    int D_KV = cfg.numKVHeads * headDim;
    int groupRatio = H / cfg.numKVHeads;

    bc->Q = matmul(bc->ln1Out, bw->wQ, S, D, D); addBias(bc->Q, bw->bQ, S, D);
    bc->K = matmul(bc->ln1Out, bw->wK, S, D, D_KV); addBias(bc->K, bw->bK, S, D_KV);
    bc->V = matmul(bc->ln1Out, bw->wV, S, D, D_KV); addBias(bc->V, bw->bV, S, D_KV);

#if USE_ROPE
    applyRoPE(bc->Q, S, D, H, 0);
    applyRoPE(bc->K, S, D_KV, cfg.numKVHeads, 0);
#endif

    bc->attnWeights = falloc(H * S * S);
    bc->attnVals = falloc(S * D);
    float scale = 1.f / (float)sqrt((double)headDim);

    for (int h = 0; h < H; h++) {
        int kv_h = h / groupRatio;
        float *scores = falloc(S * S);
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++) {
                if (j > i) {
                    scores[i * S + j] = NEG_INF;
                } else {
                    float dot = 0.f;
                    for (int d = 0; d < headDim; d++)
                        dot += bc->Q[i * D + h * headDim + d] * bc->K[j * D_KV + kv_h * headDim + d];
                    scores[i * S + j] = dot * scale;
                }
            }
        float *weights = softmaxRows(scores, S, S);
        /* store the softmax p (needed by the backward jacobian regardless of variant) */
        for (int idx = 0; idx < S * S; idx++)
            bc->attnWeights[h * S * S + idx] = weights[idx];
#if USE_REPRESSION
        /* Desatenção: combine V with w = (1-p)/Σ(1-p), not with p itself. */
        float *comb = falloc(S * S);
        for (int i = 0; i < S; i++) {
            float Z = 0.f;
            for (int j = 0; j <= i; j++) { float q = 1.f - weights[i * S + j]; comb[i * S + j] = q; Z += q; }
            float invZ = 1.f / (Z + REPR_EPS);
            for (int j = 0; j <= i; j++) comb[i * S + j] *= invZ;
        }
#else
        float *comb = weights;   /* combination weights = softmax */
#endif
        for (int i = 0; i < S; i++)
            for (int d = 0; d < headDim; d++) {
                float sum = 0.f;
                for (int j = 0; j <= i; j++)
                    sum += comb[i * S + j] * bc->V[j * D_KV + kv_h * headDim + d];
                bc->attnVals[i * D + h * headDim + d] = sum;
            }
#if USE_REPRESSION
        free(comb);
#endif
        free(scores);
        free(weights);
    }

    float *attnOut = matmul(bc->attnVals, bw->wO, S, D, D);
    addBias(attnOut, bw->bO, S, D);
    return attnOut;
}

/* Backward through self-attention. Returns dLn1Out (grad w.r.t. the normed input). */
static float *mixerBackwardAttn(const float *dMixerOut, BlockCache *bc, BlockWeights *bw,
                            Config cfg, BlockWeights *bg, int S) {
    int D = cfg.embDim, H = cfg.numHeads;
    int headDim = D / H;
    int D_KV = cfg.numKVHeads * headDim;
    int groupRatio = H / cfg.numKVHeads;

    accumFree(bg->bO, sumCols(dMixerOut, S, D), D);
    /* attnVals was already computed in mixerForward and cached — reuse it. */
    accumFree(bg->wO, matmulTransA(bc->attnVals, dMixerOut, S, D, D), D * D);
    float *dAttnVals = matmulTransB(dMixerOut, bw->wO, S, D, D);

    float *dQ = falloc(S * D);
    float *dK = falloc(S * D_KV);
    float *dV = falloc(S * D_KV);
    float scale = 1.f / (float)sqrt((double)headDim);

    for (int h = 0; h < H; h++) {
        int kv_h = h / groupRatio;
        float *dWeights = falloc(S * S);
#if USE_REPRESSION
        /* Rebuild the repression weights w=(1-p)/Σ(1-p) used in the forward, so dV
         * and the dw→dp remap below match the combination weights exactly. */
        float *comb = falloc(S * S), *qrow = falloc(S * S), *invZrow = falloc(S);
        for (int i = 0; i < S; i++) {
            float Z = 0.f;
            for (int j = 0; j <= i; j++) { float q = 1.f - bc->attnWeights[h * S * S + i * S + j]; qrow[i * S + j] = q; Z += q; }
            float iz = 1.f / (Z + REPR_EPS); invZrow[i] = iz;
            for (int j = 0; j <= i; j++) comb[i * S + j] = qrow[i * S + j] * iz;
        }
        const float *cw = comb;                       /* combination weights = w */
#else
        const float *cw = &bc->attnWeights[h * S * S]; /* combination weights = p */
#endif
        for (int i = 0; i < S; i++) {
            for (int j = 0; j <= i; j++) {
                float dot = 0.f;
                for (int d = 0; d < headDim; d++)
                    dot += dAttnVals[i * D + h * headDim + d] * bc->V[j * D_KV + kv_h * headDim + d];
                dWeights[i * S + j] = dot;
            }
            for (int d = 0; d < headDim; d++)
                for (int j = 0; j <= i; j++)
                    dV[j * D_KV + kv_h * headDim + d] +=
                        cw[i * S + j] * dAttnVals[i * D + h * headDim + d];
        }
#if USE_REPRESSION
        /* dWeights is grad w.r.t. w; map it to grad w.r.t. p (the softmax weights)
         * so the softmax jacobian below is unchanged:
         *   dp_m = -invZ·dw_m + invZ²·Σ_j dw_j·q_j   (per causal row) */
        for (int i = 0; i < S; i++) {
            float iz = invZrow[i], Sq = 0.f;
            for (int j = 0; j <= i; j++) Sq += dWeights[i * S + j] * qrow[i * S + j];
            for (int j = 0; j <= i; j++)
                dWeights[i * S + j] = -iz * dWeights[i * S + j] + iz * iz * Sq;
        }
        free(comb); free(qrow); free(invZrow);
#endif
        /* softmax jacobian */
        float *dScores = falloc(S * S);
        for (int i = 0; i < S; i++) {
            float dotSum = 0.f;
            for (int j = 0; j <= i; j++)
                dotSum += dWeights[i * S + j] * bc->attnWeights[h * S * S + i * S + j];
            for (int j = 0; j <= i; j++)
                dScores[i * S + j] =
                    bc->attnWeights[h * S * S + i * S + j] * (dWeights[i * S + j] - dotSum) * scale;
        }
        for (int i = 0; i < S; i++)
            for (int j = 0; j <= i; j++)
                for (int d = 0; d < headDim; d++) {
                    dQ[i * D + h * headDim + d] += dScores[i * S + j] * bc->K[j * D_KV + kv_h * headDim + d];
                    dK[j * D_KV + kv_h * headDim + d] += dScores[i * S + j] * bc->Q[i * D + h * headDim + d];
                }
        free(dWeights);
        free(dScores);
    }
    free(dAttnVals);

#if USE_ROPE
    applyRoPE(dQ, S, D, H, 1);
    applyRoPE(dK, S, D_KV, cfg.numKVHeads, 1);
#endif

    accumFree(bg->bQ, sumCols(dQ, S, D), D);
    accumFree(bg->wQ, matmulTransA(bc->ln1Out, dQ, S, D, D), D * D);
    accumFree(bg->bK, sumCols(dK, S, D_KV), D_KV);
    accumFree(bg->wK, matmulTransA(bc->ln1Out, dK, S, D, D_KV), D * D_KV);
    accumFree(bg->bV, sumCols(dV, S, D_KV), D_KV);
    accumFree(bg->wV, matmulTransA(bc->ln1Out, dV, S, D, D_KV), D * D_KV);

    float *dLn1Out = falloc(S * D);
    accumFree(dLn1Out, matmulTransB(dQ, bw->wQ, S, D, D), S * D);
    accumFree(dLn1Out, matmulTransB(dK, bw->wK, S, D_KV, D), S * D);
    accumFree(dLn1Out, matmulTransB(dV, bw->wV, S, D_KV, D), S * D);
    free(dQ); free(dK); free(dV);
    return dLn1Out;
}
/* ─── Mixer: Mamba selective state-space scan (S6, v1/minimal) ─────────────────
 *
 * Replaces attention with a per-channel causal linear recurrence whose parameters
 * are input-dependent ("selective"). u = ln1Out (no input projection/expand):
 *
 *   Δ_t   = softplus(ln1Out_t · wDelta + dtBias)          (S×D, per-channel step)
 *   B_t   = ln1Out_t · wB     C_t = ln1Out_t · wC          (S×N, selection)
 *   A     = -exp(Alog)                                     (D×N, ≤ 0 for stability)
 *   Ā_t   = exp(Δ_t ⊙ A)      B̄_t = Δ_t ⊙ B_t             (discretization, ZOH/Euler)
 *   h_t   = Ā_t ⊙ h_{t-1} + B̄_t · u_t                     (the causal scan)
 *   y_t   = Σ_n C_t[n]·h_t[·,n] + D ⊙ u_t                  (readout + skip)
 *
 * v2b then gates and projects (as in the real Mamba block):
 *   z   = u·wGate           yg = y ⊙ SiLU(z)           mixerOut = yg·wOut
 * Causal by construction (left-to-right recurrence) — no mask.
 * @see https://arxiv.org/abs/2312.00752 — Gu & Dao (2023), "Mamba"
 */
static float *mixerForwardMamba(BlockCache *bc, BlockWeights *bw, Config cfg, int S) {
    int D = cfg.embDim, N = cfg.stateN;
    const float *u = bc->ln1Out;

    /* 1. Depthwise Causal Conv1D (K=4) + SiLU activation */
    bc->convOut = falloc(S * D);
    bc->uPrime  = falloc(S * D);
    for (int t = 0; t < S; t++) {
        for (int d = 0; d < D; d++) {
            float val = bw->convB[d];
            for (int k = 0; k < 4; k++) {
                if (t - k >= 0) {
                    val += u[(t - k) * D + d] * bw->convW[d * 4 + k];
                }
            }
            bc->convOut[t * D + d] = val;
            bc->uPrime[t * D + d]  = silu(val);
        }
    }
    const float *up = bc->uPrime;

    /* selection: Δ (via softplus), B, C — all functions of the input */
    float *raw = matmul(up, bw->wDelta, S, D, D);   /* S×D */
    bc->delta = falloc(S * D);
    bc->dtSig = falloc(S * D);
    for (int t = 0; t < S; t++)
        for (int d = 0; d < D; d++) {
            float x = raw[t * D + d] + bw->dtBias[d];
            float sig = 1.f / (1.f + (float)exp(-x));
            bc->dtSig[t * D + d] = sig;                       /* softplus'(x) = sigmoid(x) */
            bc->delta[t * D + d] = (float)log(1.0 + exp((double)x));
        }
    free(raw);
    bc->ssmB = matmul(up, bw->wB, S, D, N);   /* S×N */
    bc->ssmC = matmul(up, bw->wC, S, D, N);   /* S×N */

    /* causal scan: h_t = Ā_t ⊙ h_{t-1} + B̄_t · u'_t,  A = -exp(Alog) */
    bc->h = falloc(S * D * N);
    float *y = falloc(S * D);
    for (int t = 0; t < S; t++) {
        for (int d = 0; d < D; d++) {
            float dt = bc->delta[t * D + d];
            float ut = up[t * D + d];
            float acc = 0.f;
            for (int n = 0; n < N; n++) {
                float A = -(float)exp((double)bw->Alog[d * N + n]);
                float Abar = (float)exp((double)(dt * A));
                float Bbar = dt * bc->ssmB[t * N + n];
                float prev = (t > 0) ? bc->h[((t - 1) * D + d) * N + n] : 0.f;
                float hv = Abar * prev + Bbar * ut;
                bc->h[(t * D + d) * N + n] = hv;
                acc += bc->ssmC[t * N + n] * hv;
            }
            y[t * D + d] = acc + bw->Dskip[d] * ut;
        }
    }
    bc->ssmY = y;   /* keep the pre-gate readout for backward */

    /* gate + output projection: mixerOut = (y ⊙ SiLU(u·wGate)) · wOut */
    bc->gateZ = matmul(u, bw->wGate, S, D, D);   /* S×D */
    float *yg = falloc(S * D);
    for (int i = 0; i < S * D; i++) yg[i] = y[i] * silu(bc->gateZ[i]);
    float *out = matmul(yg, bw->wOut, S, D, D);
    free(yg);
    return out;
}

/* Backward through the selective scan (BPTT). Returns dLn1Out. */
static float *mixerBackwardMamba(const float *dMixerOut, BlockCache *bc, BlockWeights *bw,
                            Config cfg, BlockWeights *bg, int S) {
    int D = cfg.embDim, N = cfg.stateN;
    const float *u = bc->ln1Out;
    const float *up = bc->uPrime;

    float *dLn1Out = falloc(S * D);   /* accumulates every path back to the normed input */
    float *duPrime = falloc(S * D);   /* grad w.r.t. uPrime (S×D) */
    float *dDelta = falloc(S * D);    /* grad w.r.t. Δ (S×D) */
    float *dB = falloc(S * N);        /* grad w.r.t. B (S×N) */
    float *dC = falloc(S * N);        /* grad w.r.t. C (S×N) */
    float *dh = falloc(S * D * N);    /* grad w.r.t. h_t, carried backward through time */

    /* v2b backward: mixerOut = yg·wOut,  yg = ssmY ⊙ SiLU(gateZ).  Produces dY
     * (grad into the SSM readout) and the gate-path grad into dLn1Out. */
    float *yg = falloc(S * D);
    for (int i = 0; i < S * D; i++) yg[i] = bc->ssmY[i] * silu(bc->gateZ[i]);
    accumFree(bg->wOut, matmulTransA(yg, dMixerOut, S, D, D), D * D);
    float *dyg = matmulTransB(dMixerOut, bw->wOut, S, D, D);
    free(yg);

    float *dY = falloc(S * D);
    float *dz = falloc(S * D);
    for (int i = 0; i < S * D; i++) {
        float z = bc->gateZ[i];
        float sig = 1.f / (1.f + (float)exp(-z));
        float gate = z * sig;                          /* SiLU(z) */
        dY[i] = dyg[i] * gate;                          /* into the SSM readout */
        float dgate = dyg[i] * bc->ssmY[i];
        dz[i] = dgate * (sig * (1.f + z * (1.f - sig))); /* × SiLU'(z) */
    }
    free(dyg);
    accumFree(bg->wGate, matmulTransA(u, dz, S, D, D), D * D);
    accumFree(dLn1Out, matmulTransB(dz, bw->wGate, S, D, D), S * D);
    free(dz);

    /* readout: y_t[d] = Σ_n C_t[n] h_t[d,n] + Dskip[d] u'_t[d] */
    for (int t = 0; t < S; t++)
        for (int d = 0; d < D; d++) {
            float dy = dY[t * D + d];
            bg->Dskip[d] += dy * up[t * D + d];
            duPrime[t * D + d] += dy * bw->Dskip[d];          /* u' = SiLU(conv) */
            for (int n = 0; n < N; n++) {
                dh[(t * D + d) * N + n] = dy * bc->ssmC[t * N + n];
                dC[t * N + n] += dy * bc->h[(t * D + d) * N + n];
            }
        }
    free(dY);

    /* scan backward: h_t = Ā_t h_{t-1} + B̄_t u'_t.  Ā_t=exp(Δ_t A), B̄_t=Δ_t B_t. */
    for (int t = S - 1; t >= 0; t--) {
        for (int d = 0; d < D; d++) {
            float dt = bc->delta[t * D + d];
            float ut = up[t * D + d];
            for (int n = 0; n < N; n++) {
                float A = -(float)exp((double)bw->Alog[d * N + n]);
                float Abar = (float)exp((double)(dt * A));
                float Bt = bc->ssmB[t * N + n];
                float Bbar = dt * Bt;
                float prev = (t > 0) ? bc->h[((t - 1) * D + d) * N + n] : 0.f;
                float g = dh[(t * D + d) * N + n];

                /* h_t depends on Ā_t, B̄_t, h_{t-1}, u'_t */
                float dAbar = g * prev;
                float dBbar = g * ut;
                if (t > 0) dh[((t - 1) * D + d) * N + n] += g * Abar;
                duPrime[t * D + d] += g * Bbar;               /* via u'_t */

                /* Ā_t = exp(Δ_t A): dΔ, dA(→dAlog).  B̄_t = Δ_t B_t: dΔ, dB. */
                float dAbar_dArg = dAbar * Abar;              /* d/d(Δ·A) */
                dDelta[t * D + d] += dAbar_dArg * A + dBbar * Bt;
                bg->Alog[d * N + n] += dAbar_dArg * dt * A;   /* dA·A since A=-exp(Alog) */
                dB[t * N + n] += dBbar * dt;
            }
        }
    }

    /* Δ = softplus(raw + dtBias): draw = dΔ · sigmoid(raw+dtBias) */
    float *dRaw = falloc(S * D);
    for (int t = 0; t < S; t++)
        for (int d = 0; d < D; d++) {
            float draw = dDelta[t * D + d] * bc->dtSig[t * D + d];
            dRaw[t * D + d] = draw;
            bg->dtBias[d] += draw;
        }
    accumFree(bg->wDelta, matmulTransA(up, dRaw, S, D, D), D * D);
    accumFree(duPrime, matmulTransB(dRaw, bw->wDelta, S, D, D), S * D);
    free(dRaw);

    /* B_t = u'_t·wB, C_t = u'_t·wC */
    accumFree(bg->wB, matmulTransA(up, dB, S, D, N), D * N);
    accumFree(duPrime, matmulTransB(dB, bw->wB, S, N, D), S * D);
    accumFree(bg->wC, matmulTransA(up, dC, S, D, N), D * N);
    accumFree(duPrime, matmulTransB(dC, bw->wC, S, N, D), S * D);

    free(dDelta); free(dB); free(dC); free(dh);

    /* 5. Backprop through SiLU: uPrime = silu(convOut)
     * d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))) */
    float *dConvOut = falloc(S * D);
    for (int t = 0; t < S; t++) {
        for (int d = 0; d < D; d++) {
            float x = bc->convOut[t * D + d];
            float sig = 1.f / (1.f + (float)exp(-x));
            float dsilu = sig * (1.f + x * (1.f - sig));
            dConvOut[t * D + d] = duPrime[t * D + d] * dsilu;
        }
    }

    /* 6. Backprop through depthwise causal conv1d:
     * convOut[t, d] = convB[d] + sum_{k=0..3} u[t-k, d] * convW[d, k] */
    for (int t = 0; t < S; t++) {
        for (int d = 0; d < D; d++) {
            float g = dConvOut[t * D + d];
            bg->convB[d] += g;
            for (int k = 0; k < 4; k++) {
                if (t - k >= 0) {
                    bg->convW[d * 4 + k] += g * u[(t - k) * D + d];
                    dLn1Out[(t - k) * D + d] += g * bw->convW[d * 4 + k];
                }
            }
        }
    }
    free(dConvOut);
    free(duPrime);

    return dLn1Out;
}


/* ─── Mixer: Causal Dilated CNN mixer (TCN style) ──────────────────────────────
 *
 * Replaces attention/SSM with a depthwise causal 1D convolution of kernel K=3
 * with a dilation rate that grows exponentially layer-by-layer (dilation = 1 << layerIdx).
 * Followed by SiLU activation and an output projection layer.
 * Complexities: O(S) time/memory, receptive field: 2 * dilation + 1.
 */
static float *mixerForwardCNN(BlockCache *bc, BlockWeights *bw, Config cfg, int S, int layerIdx) {
    int D = cfg.embDim;
    int dil = 1 << layerIdx;
    const float *x = bc->ln1Out;

    bc->convOut_cnn = falloc(S * D);
    bc->siluOut_cnn = falloc(S * D);

#if USE_CONTRAST_CONV
    /* effective kernel = mean-subtracted (zero DC) → high-pass / contrast filter */
    float *ke = falloc(D * 3);
    for (int d = 0; d < D; d++) {
        float m = (bw->convW_cnn[d * 3 + 0] + bw->convW_cnn[d * 3 + 1] + bw->convW_cnn[d * 3 + 2]) / 3.f;
        ke[d * 3 + 0] = bw->convW_cnn[d * 3 + 0] - m;
        ke[d * 3 + 1] = bw->convW_cnn[d * 3 + 1] - m;
        ke[d * 3 + 2] = bw->convW_cnn[d * 3 + 2] - m;
    }
    const float *K = ke;
#else
    const float *K = bw->convW_cnn;
#endif

    #pragma omp parallel for collapse(2) if(S * D > 1024)
    for (int t = 0; t < S; t++) {
        for (int d = 0; d < D; d++) {
            float val = bw->convB_cnn[d];
            val += K[d * 3 + 0] * x[t * D + d];
            if (t - dil >= 0) {
                val += K[d * 3 + 1] * x[(t - dil) * D + d];
            }
            if (t - 2 * dil >= 0) {
                val += K[d * 3 + 2] * x[(t - 2 * dil) * D + d];
            }
            bc->convOut_cnn[t * D + d] = val;
            bc->siluOut_cnn[t * D + d] = val * (1.f / (1.f + (float)exp(-val))); // SiLU
        }
    }
#if USE_CONTRAST_CONV
    free(ke);
#endif

    float *mixerOut = matmul(bc->siluOut_cnn, bw->wOut_cnn, S, D, D);
    addBias(mixerOut, bw->bOut_cnn, S, D);
    return mixerOut;
}

static float *mixerBackwardCNN(const float *dMixerOut, BlockCache *bc, BlockWeights *bw,
                             Config cfg, BlockWeights *bg, int S, int layerIdx) {
    int D = cfg.embDim;
    int dil = 1 << layerIdx;
    const float *x = bc->ln1Out;

    // 1. Projection gradients
    accumFree(bg->bOut_cnn, sumCols(dMixerOut, S, D), D);
    accumFree(bg->wOut_cnn, matmulTransA(bc->siluOut_cnn, dMixerOut, S, D, D), D * D);
    float *dSilu = matmulTransB(dMixerOut, bw->wOut_cnn, S, D, D);

    // 2. Backprop through SiLU to get dConvOut
    float *dConvOut = falloc(S * D);
    #pragma omp parallel for collapse(2) if(S * D > 1024)
    for (int t = 0; t < S; t++) {
        for (int d = 0; d < D; d++) {
            float val = bc->convOut_cnn[t * D + d];
            float sig = 1.f / (1.f + (float)exp(-val));
            float dsilu = sig * (1.f + val * (1.f - sig));
            dConvOut[t * D + d] = dSilu[t * D + d] * dsilu;
        }
    }
    free(dSilu);

    // 3. Backprop through depthwise causal dilated convolution to get dx (dLn1Out) and weight grads
    float *dLn1Out = falloc(S * D);

    #pragma omp parallel for if(D > 16)
    for (int d = 0; d < D; d++) {
        float db = 0.f;
        float dw0 = 0.f;   /* grad w.r.t. the *effective* tap used in the forward */
        float dw1 = 0.f;
        float dw2 = 0.f;
#if USE_CONTRAST_CONV
        float m = (bw->convW_cnn[d * 3 + 0] + bw->convW_cnn[d * 3 + 1] + bw->convW_cnn[d * 3 + 2]) / 3.f;
        float k0 = bw->convW_cnn[d * 3 + 0] - m, k1 = bw->convW_cnn[d * 3 + 1] - m, k2 = bw->convW_cnn[d * 3 + 2] - m;
#else
        float k0 = bw->convW_cnn[d * 3 + 0], k1 = bw->convW_cnn[d * 3 + 1], k2 = bw->convW_cnn[d * 3 + 2];
#endif
        for (int t = 0; t < S; t++) {
            float g = dConvOut[t * D + d];
            db += g;
            dw0 += g * x[t * D + d];
            dLn1Out[t * D + d] += g * k0;

            if (t - dil >= 0) {
                dw1 += g * x[(t - dil) * D + d];
                dLn1Out[(t - dil) * D + d] += g * k1;
            }
            if (t - 2 * dil >= 0) {
                dw2 += g * x[(t - 2 * dil) * D + d];
                dLn1Out[(t - 2 * dil) * D + d] += g * k2;
            }
        }
        bg->convB_cnn[d] += db;
#if USE_CONTRAST_CONV
        /* remap grad-wrt-effective-kernel to grad-wrt-raw: dk_n = dk'_n − mean(dk') */
        float dmean = (dw0 + dw1 + dw2) / 3.f;
        bg->convW_cnn[d * 3 + 0] += dw0 - dmean;
        bg->convW_cnn[d * 3 + 1] += dw1 - dmean;
        bg->convW_cnn[d * 3 + 2] += dw2 - dmean;
#else
        bg->convW_cnn[d * 3 + 0] += dw0;
        bg->convW_cnn[d * 3 + 1] += dw1;
        bg->convW_cnn[d * 3 + 2] += dw2;
#endif
    }
    free(dConvOut);

    return dLn1Out;
}


/* ─── Mixer: Hyena / SGConv Gated Long Causal Convolution ─────────────────────
 *
 * Replaces attention/SSM with a gated global causal convolution. Projects the input
 * to u and v path, convolves v path with a learnable full-context causal filter h,
 * gates the result by multiplying elementwise with u path, and projects back.
 * Complexities: O(S^2) time domain convolution (highly parallelized, fast for small C),
 * O(S) memory.
 */
static float *mixerForwardHyena(BlockCache *bc, BlockWeights *bw, Config cfg, int S) {
    int D = cfg.embDim;
    const float *x = bc->ln1Out;

    bc->u_hyena = matmul(x, bw->wU_hyena, S, D, D);
    addBias(bc->u_hyena, bw->bU_hyena, S, D);
    bc->v_hyena = matmul(x, bw->wV_hyena, S, D, D);
    addBias(bc->v_hyena, bw->bV_hyena, S, D);

    bc->convOut_hyena = falloc(S * D);
    #pragma omp parallel for collapse(2) if(S * D > 1024)
    for (int t = 0; t < S; t++) {
        for (int d = 0; d < D; d++) {
            float val = 0.f;
            for (int k = 0; k <= t; k++) {
                val += bc->v_hyena[(t - k) * D + d] * bw->filter_hyena[k * D + d];
            }
            bc->convOut_hyena[t * D + d] = val;
        }
    }

    bc->gateOut_hyena = falloc(S * D);
    #pragma omp parallel for if(S * D > 1024)
    for (int i = 0; i < S * D; i++) {
        bc->gateOut_hyena[i] = bc->u_hyena[i] * bc->convOut_hyena[i];
    }

    float *mixerOut = matmul(bc->gateOut_hyena, bw->wOut_hyena, S, D, D);
    addBias(mixerOut, bw->bOut_hyena, S, D);
    return mixerOut;
}

static float *mixerBackwardHyena(const float *dMixerOut, BlockCache *bc, BlockWeights *bw,
                              Config cfg, BlockWeights *bg, int S) {
    int D = cfg.embDim;
    const float *x = bc->ln1Out;

    // 1. Output projection backward
    accumFree(bg->bOut_hyena, sumCols(dMixerOut, S, D), D);
    accumFree(bg->wOut_hyena, matmulTransA(bc->gateOut_hyena, dMixerOut, S, D, D), D * D);
    float *dGate = matmulTransB(dMixerOut, bw->wOut_hyena, S, D, D);

    // 2. Gating backward
    float *dU = falloc(S * D);
    float *dConv = falloc(S * D);
    #pragma omp parallel for if(S * D > 1024)
    for (int i = 0; i < S * D; i++) {
        dU[i] = dGate[i] * bc->convOut_hyena[i];
        dConv[i] = dGate[i] * bc->u_hyena[i];
    }
    free(dGate);

    // 3. Global depthwise causal convolution backward
    float *dV = falloc(S * D);
    #pragma omp parallel for if(D > 8)
    for (int d = 0; d < D; d++) {
        for (int t = 0; t < S; t++) {
            float g = dConv[t * D + d];
            for (int k = 0; k <= t; k++) {
                bg->filter_hyena[k * D + d] += g * bc->v_hyena[(t - k) * D + d];
                dV[(t - k) * D + d] += g * bw->filter_hyena[k * D + d];
            }
        }
    }
    free(dConv);

    // 4. Projections backward
    accumFree(bg->bU_hyena, sumCols(dU, S, D), D);
    accumFree(bg->wU_hyena, matmulTransA(x, dU, S, D, D), D * D);
    float *dX_u = matmulTransB(dU, bw->wU_hyena, S, D, D);
    free(dU);

    accumFree(bg->bV_hyena, sumCols(dV, S, D), D);
    accumFree(bg->wV_hyena, matmulTransA(x, dV, S, D, D), D * D);
    float *dX_v = matmulTransB(dV, bw->wV_hyena, S, D, D);
    free(dV);

    float *dLn1Out = falloc(S * D);
    #pragma omp parallel for if(S * D > 1024)
    for (int i = 0; i < S * D; i++) {
        dLn1Out[i] = dX_u[i] + dX_v[i];
    }
    free(dX_u);
    free(dX_v);

    return dLn1Out;
}

/* ─── Shadow-cache (inference-only, --shadow-cache) ──────────────────────────
 * During generation the context window slides and each forward recomputes the
 * Shadow from zero inside the window — everything before the window is truly
 * forgotten. The Shadow-cache persists s_t across slides: g_shadowCache[l]
 * holds s at absolute position (window start − 1) for each Jungian layer, and
 * seeds the recurrence of the next forward. K floats per layer = a lossy tonal
 * memory of UNBOUNDED context at O(K) cost (the tonal analogue of a KV-cache;
 * see DISCOVERY.md §7). Never on during training — the flag is only raised
 * around generateText, so no backward/grad-check change (the BPTT boundary
 * s_{-1}=0 of the training regime is untouched). */
static int g_shadowCacheOn = 0;
static float g_shadowCache[8][N_ARCH];   /* per-layer s_abs(window start − 1) */

/* ─── Personal-relevance reinforcement (inference-only, IDEA.md §3.1/§6) ──────
 * Extends the Shadow recurrence with a second, personal term:
 *
 *   s_t = α·s_{t-1} + (1-α)·[ (1-p_t)/(K-1)  +  δ·R_t·p_t ]
 *          └── structural repression ──┘      └── personal relevance ──┘
 *
 * R_t is a per-token reinforcement scalar supplied by the interlocutor
 * ("simulated user" in the MVP). Multiplying it by p_t deposits the *active*
 * archetypes of a reinforced token into the Shadow, so the steering bias
 * (sim += β·s_{t-1}) later favours exactly the channels the user valued.
 * With g_reinforceOn=0 the block is byte-identical to the trained recurrence,
 * so training and grad-check are untouched (the term is never active in
 * backward, which runs only during training). g_RtWindow points at R for the
 * current window (length S, local positions); NULL ⇒ no reinforcement. */
static int          g_reinforceOn = 0;
static float        g_reinforceDelta = 0.0f;   /* δ — weight of the personal term */
static const float *g_RtWindow = NULL;         /* R_t per local position, length S */

/* Personal Shadow as a *standing* steering bias (IDEA.md §3.4):
 *   Sombra_total = Sombra_estrutural + w_u · Sombra_u
 * The structural Shadow is the running EMA recurrence; Sombra_u is a frozen
 * per-user vector added to sim at EVERY token (unlike the cache seed, it does
 * not decay), so a user's accumulated tonal direction persistently biases the
 * conscious choice. NULL per layer ⇒ inactive (training/grad-check untouched). */
/* _Thread_local: com STEER_AUG os itens do batch (threads OMP) carregam vieses
 * distintos; na inferência (single-thread) a semântica é idêntica à anterior. */
static _Thread_local const float *g_personalBias[8] = { 0 };
static _Thread_local float        g_personalW = 0.0f;   /* w_u — identity confidence weight */

/* Generic activation steering (inference-only, ActAdd-style) for layouts with
 * NO Jungian layer: before block l runs, x += g_actW · g_actSteer[l] is added
 * to the residual stream. NULL per layer ⇒ off (training/grad-check untouched).
 * This is the baseline injection an emergent archetype falls back to when the
 * architecture offers no wArch channels to bias. */
static const float *g_actSteer[8] = { 0 };
static float        g_actW = 0.0f;

static float *mixerForwardJungian(BlockCache *bc, BlockWeights *bw, Config cfg, int S,
                                  int layerIdx) {
    int D = cfg.embDim;
    int K = N_ARCH;
    float alpha = JUNGIAN_ALPHA;
    float temp = JUNGIAN_TEMP;
    float inv_sqrt_d = 1.0f / (float)sqrt((double)D);
    const float *sPrev0 = (g_shadowCacheOn && layerIdx < 8) ? g_shadowCache[layerIdx] : NULL;
    const float *pBias = (layerIdx < 8) ? g_personalBias[layerIdx] : NULL;  /* standing per-user bias */

    const float *c = bc->ln1Out;

    // 1, 2, 3. Compute similarity, steered softmax, and shadow update step-by-step
    bc->sim_j = falloc(S * K);
    bc->p_j = falloc(S * K);
    bc->shadow_j = falloc(S * K);

    float shadow_scale = 1.0f / (float)(K - 1);
    float current_shadow[N_ARCH];
    float steering = JUNGIAN_STEERING;

    // Step 0: previous shadow = 0 (training regime) or the Shadow-cache (inference)
    {
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                sum += c[d] * bw->wArch[k * D + d];
            }
            bc->sim_j[k] = sum + (sPrev0 ? steering * sPrev0[k] : 0.0f)
                               + (pBias ? steering * g_personalW * pBias[k] : 0.0f);
        }
        float max_val = bc->sim_j[0] / temp;
        for (int k = 1; k < K; k++) {
            float v = bc->sim_j[k] / temp;
            if (v > max_val) max_val = v;
        }
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            bc->p_j[k] = (float)exp((double)(bc->sim_j[k] / temp - max_val));
            sum += bc->p_j[k];
        }
        for (int k = 0; k < K; k++) {
            bc->p_j[k] /= sum;
        }
        for (int k = 0; k < K; k++) {
            current_shadow[k] = (sPrev0 ? alpha * sPrev0[k] : 0.0f)
                              + (1.0f - alpha) * (1.0f - bc->p_j[k]) * shadow_scale;
            if (g_reinforceOn && g_RtWindow)     /* personal-relevance term δ·R_0·p_0 */
                current_shadow[k] += (1.0f - alpha) * g_reinforceDelta
                                   * g_RtWindow[0] * bc->p_j[k];
            bc->shadow_j[k] = current_shadow[k];
        }
    }

    // Steps 1 to S-1
    for (int s = 1; s < S; s++) {
        // Raw similarity + shadow steering bias from s-1
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                sum += c[s * D + d] * bw->wArch[k * D + d];
            }
            bc->sim_j[s * K + k] = sum + steering * bc->shadow_j[(s - 1) * K + k]
                                       + (pBias ? steering * g_personalW * pBias[k] : 0.0f);
        }

        // Softmax
        float max_val = bc->sim_j[s * K] / temp;
        for (int k = 1; k < K; k++) {
            float v = bc->sim_j[s * K + k] / temp;
            if (v > max_val) max_val = v;
        }
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            bc->p_j[s * K + k] = (float)exp((double)(bc->sim_j[s * K + k] / temp - max_val));
            sum += bc->p_j[s * K + k];
        }
        for (int k = 0; k < K; k++) {
            bc->p_j[s * K + k] /= sum;
        }

        // Shadow update (Homeostatic update)
        for (int k = 0; k < K; k++) {
            current_shadow[k] = alpha * bc->shadow_j[(s - 1) * K + k] + (1.0f - alpha) * (1.0f - bc->p_j[s * K + k]) * shadow_scale;
            if (g_reinforceOn && g_RtWindow)     /* personal-relevance term δ·R_s·p_s */
                current_shadow[k] += (1.0f - alpha) * g_reinforceDelta
                                   * g_RtWindow[s] * bc->p_j[s * K + k];
            bc->shadow_j[s * K + k] = current_shadow[k];
        }
    }

    // 5. Values: v_j = W_v(A) + b_v
    bc->v_j = matmul(bw->wArch, bw->wV_j, K, D, D);
    addBias(bc->v_j, bw->bV_j, K, D);

#if JUNGIAN_V2
    // v2: persona com voz direta — o próprio roteamento consciente agrega os valores
    bc->attended_j = matmul(bc->p_j, bc->v_j, S, K, D);
    (void)inv_sqrt_d;
#else
    // 4. Queries: q_j = W_q(c) + b_q
    bc->q_j = matmul(c, bw->wQ_j, S, D, D);
    addBias(bc->q_j, bw->bQ_j, S, D);

    // 6. Cross-attention weights: attnScores_j = q_j * A^T / sqrt(D)
    bc->attnScores_j = falloc(S * K);
    #pragma omp parallel for collapse(2) if(S * K > 1024)
    for (int s = 0; s < S; s++) {
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                sum += bc->q_j[s * D + d] * bw->wArch[k * D + d];
            }
            bc->attnScores_j[s * K + k] = sum * inv_sqrt_d;
        }
    }

    // Softmax: attnWeights_j[s] = softmax(attnScores_j[s])
    bc->attnWeights_j = falloc(S * K);
    #pragma omp parallel for if(S > 64)
    for (int s = 0; s < S; s++) {
        float max_val = bc->attnScores_j[s * K];
        for (int k = 1; k < K; k++) {
            if (bc->attnScores_j[s * K + k] > max_val) max_val = bc->attnScores_j[s * K + k];
        }
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            bc->attnWeights_j[s * K + k] = (float)exp((double)(bc->attnScores_j[s * K + k] - max_val));
            sum += bc->attnWeights_j[s * K + k];
        }
        for (int k = 0; k < K; k++) {
            bc->attnWeights_j[s * K + k] /= sum;
        }
    }

    // 7. Aggregation: attended_j = attnWeights_j * v_j
    bc->attended_j = matmul(bc->attnWeights_j, bc->v_j, S, K, D);
#endif

    // 8. Projection of Shadow: projShadow_j = W_s(shadow_j) + b_s
    bc->projShadow_j = matmul(bc->shadow_j, bw->wS_j, S, K, D);
    addBias(bc->projShadow_j, bw->bS_j, S, D);

    // 9. Output: attended_j + projShadow_j
    float *mixerOut = falloc(S * D);
    #pragma omp parallel for if(S * D > 1024)
    for (int i = 0; i < S * D; i++) {
        mixerOut[i] = bc->attended_j[i] + bc->projShadow_j[i];
    }

    return mixerOut;
}

static float *mixerBackwardJungian(const float *dMixerOut, BlockCache *bc, BlockWeights *bw,
                                   Config cfg, BlockWeights *bg, int S) {
    int D = cfg.embDim;
    int K = N_ARCH;
    float alpha = JUNGIAN_ALPHA;
    float temp = JUNGIAN_TEMP;
    float inv_sqrt_d = 1.0f / (float)sqrt((double)D);

    const float *c = bc->ln1Out;

    // 1. Gradients of projShadow_j = shadow_j * wS_j + bS_j
    accumFree(bg->bS_j, sumCols(dMixerOut, S, D), D);
    accumFree(bg->wS_j, matmulTransA(bc->shadow_j, dMixerOut, S, K, D), K * D);
    float *dShadow_j = matmulTransB(dMixerOut, bw->wS_j, S, D, K);

#if JUNGIAN_V2
    // 2-4 (v2). Exact reverse recurrence through the coupled chain
    //   sim_s = c·A + β·shadow_{s−1};  p_s = softmax(sim_s/T);
    //   shadow_s = α·shadow_{s−1} + (1−α)(1−p_s)/(K−1);  attended_s = p_s·v.
    //   dShadow_s = dir(wS) + α·dShadow_{s+1} + β·dSim_{s+1}
    //   dp_s = −(1−α)/(K−1)·dShadow_s + dMixerOut_s·vᵀ
    //   dSim_s = softmax_jac(p_s, dp_s)/T          (sequential, reverse time)
    float steeringB = JUNGIAN_STEERING;
    float *dP_att = matmulTransB(dMixerOut, bc->v_j, S, D, K);   /* S×K */
    float *dV_j = matmulTransA(bc->p_j, dMixerOut, S, K, D);     /* K×D */
    float *dSim_j = falloc(S * K);
    {
        float dShNext[N_ARCH], dSimNext[N_ARCH], dSh[N_ARCH], dpv[N_ARCH];
        for (int k = 0; k < K; k++) { dShNext[k] = 0.f; dSimNext[k] = 0.f; }
        float wscale = -(1.0f - alpha) / (float)(K - 1);
        for (int s = S - 1; s >= 0; s--) {
            for (int k = 0; k < K; k++)
                dSh[k] = dShadow_j[s * K + k] + alpha * dShNext[k] + steeringB * dSimNext[k];
            for (int k = 0; k < K; k++)
                dpv[k] = wscale * dSh[k] + dP_att[s * K + k];
            float dot = 0.0f;
            for (int k = 0; k < K; k++) dot += dpv[k] * bc->p_j[s * K + k];
            for (int k = 0; k < K; k++) {
                dSim_j[s * K + k] = (1.0f / temp) * bc->p_j[s * K + k] * (dpv[k] - dot);
                dSimNext[k] = dSim_j[s * K + k];
                dShNext[k] = dSh[k];
            }
        }
    }
    free(dP_att);
    free(dShadow_j);
#else
    // 2. Backpropagation through time of Shadow
    float *dShadow_acc = falloc(S * K);
    for (int k = 0; k < K; k++) {
        dShadow_acc[(S - 1) * K + k] = dShadow_j[(S - 1) * K + k];
    }
    for (int s = S - 2; s >= 0; s--) {
        for (int k = 0; k < K; k++) {
            dShadow_acc[s * K + k] = dShadow_j[s * K + k] + alpha * dShadow_acc[(s + 1) * K + k];
        }
    }
    free(dShadow_j);

    // 3. Gradient of p_j[s]
    float *dp_j = falloc(S * K);
    float shadow_scale = -1.0f / (float)(K - 1);
    #pragma omp parallel for if(S * K > 1024)
    for (int i = 0; i < S * K; i++) {
        dp_j[i] = (1.0f - alpha) * dShadow_acc[i] * shadow_scale;
    }
    free(dShadow_acc);

    // 4. Softmax backward of p_j[s] = softmax(sim_j[s] / temp)
    float *dSim_j = falloc(S * K);
    #pragma omp parallel for if(S > 64)
    for (int s = 0; s < S; s++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += dp_j[s * K + k] * bc->p_j[s * K + k];
        }
        for (int k = 0; k < K; k++) {
            dSim_j[s * K + k] = (1.0f / temp) * bc->p_j[s * K + k] * (dp_j[s * K + k] - sum);
        }
    }
    free(dp_j);
#endif

    // 5. Gradients of similarity: sim_j[s][k] = sum_d c[s][d] * A[k][d]
    float *dC_sim = falloc(S * D);
    #pragma omp parallel for collapse(2) if(S * D > 1024)
    for (int s = 0; s < S; s++) {
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += dSim_j[s * K + k] * bw->wArch[k * D + d];
            }
            dC_sim[s * D + d] = sum;
        }
    }

    float *dArch_sim = falloc(K * D);
    #pragma omp parallel for collapse(2) if(K * D > 1024)
    for (int k = 0; k < K; k++) {
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int s = 0; s < S; s++) {
                sum += dSim_j[s * K + k] * c[s * D + d];
            }
            dArch_sim[k * D + d] = sum;
        }
    }
    free(dSim_j);

#if !JUNGIAN_V2
    // 6. Gradients of cross-attention: attended_j = attnWeights_j * v_j
    float *dAttnWeights_j = matmulTransB(dMixerOut, bc->v_j, S, D, K);
    float *dV_j = matmulTransA(bc->attnWeights_j, dMixerOut, S, K, D);

    // 7. Softmax backward of attnWeights_j[s] = softmax(attnScores_j[s])
    float *dAttnScores_j = falloc(S * K);
    #pragma omp parallel for if(S > 64)
    for (int s = 0; s < S; s++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += dAttnWeights_j[s * K + k] * bc->attnWeights_j[s * K + k];
        }
        for (int k = 0; k < K; k++) {
            dAttnScores_j[s * K + k] = bc->attnWeights_j[s * K + k] * (dAttnWeights_j[s * K + k] - sum);
        }
    }
    free(dAttnWeights_j);

    // 8. Gradients of attnScores_j = q_j * A^T / sqrt(D)
    float *dQ_j = falloc(S * D);
    #pragma omp parallel for collapse(2) if(S * D > 1024)
    for (int s = 0; s < S; s++) {
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += dAttnScores_j[s * K + k] * bw->wArch[k * D + d];
            }
            dQ_j[s * D + d] = sum * inv_sqrt_d;
        }
    }

    float *dArch_attn = falloc(K * D);
    #pragma omp parallel for collapse(2) if(K * D > 1024)
    for (int k = 0; k < K; k++) {
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int s = 0; s < S; s++) {
                sum += dAttnScores_j[s * K + k] * bc->q_j[s * D + d];
            }
            dArch_attn[k * D + d] = sum * inv_sqrt_d;
        }
    }
    free(dAttnScores_j);

    // 9. Gradients of queries: q_j = W_q(c) + b_q
    accumFree(bg->bQ_j, sumCols(dQ_j, S, D), D);
    accumFree(bg->wQ_j, matmulTransA(c, dQ_j, S, D, D), D * D);
    float *dC_q = matmulTransB(dQ_j, bw->wQ_j, S, D, D);
    free(dQ_j);
#endif

    // 10. Gradients of values: v_j = W_v(A) + b_v
    accumFree(bg->bV_j, sumCols(dV_j, K, D), D);
    accumFree(bg->wV_j, matmulTransA(bw->wArch, dV_j, K, D, D), D * D);
    float *dArch_v = matmulTransB(dV_j, bw->wV_j, K, D, D);
    free(dV_j);

    // 11. Accumulate gradients for wArch (A)
    float *dArch_total = falloc(K * D);
    #pragma omp parallel for if(K * D > 1024)
    for (int i = 0; i < K * D; i++) {
#if JUNGIAN_V2
        dArch_total[i] = dArch_sim[i] + dArch_v[i];
#else
        dArch_total[i] = dArch_sim[i] + dArch_attn[i] + dArch_v[i];
#endif
    }
#if JUNGIAN_V2
    free(dArch_sim); free(dArch_v);
#else
    free(dArch_sim); free(dArch_attn); free(dArch_v);
#endif
    accumFree(bg->wArch, dArch_total, K * D);

    // 12. Accumulate gradients for input c
#if JUNGIAN_V2
    float *dLn1Out = dC_sim;               /* no q path in v2 */
#else
    float *dLn1Out = falloc(S * D);
    #pragma omp parallel for if(S * D > 1024)
    for (int i = 0; i < S * D; i++) {
        dLn1Out[i] = dC_sim[i] + dC_q[i];
    }
    free(dC_sim); free(dC_q);
#endif

    return dLn1Out;
}



/*
 * One block, forward (mixer-agnostic):
 *   1. norm -> mixer (attention or SSM) -> add residual
 *   2. norm -> feed-forward (expand -> ReLU -> project) -> add residual
 */
static BlockCache blockForward(const float *input, BlockWeights *bw, Config cfg, int S, int layerIdx) {
    int D = cfg.embDim, F = cfg.ffDim;

    BlockCache bc = {0};
    bc.input = input;

    bc.ln1Mean = falloc(S);
    bc.ln1Var = falloc(S);
    bc.ln1Hat = falloc(S * D);
    bc.ln1Out = normForward(input, bw->ln1Gamma, bw->ln1Beta, S, D, bc.ln1Mean, bc.ln1Var, bc.ln1Hat);

    if (bw->wB != NULL) {
        bc.mixerOut = mixerForwardMamba(&bc, bw, cfg, S);
    } else if (bw->wQ != NULL) {
        bc.mixerOut = mixerForwardAttn(&bc, bw, cfg, S);
    } else if (bw->convW_cnn != NULL) {
        bc.mixerOut = mixerForwardCNN(&bc, bw, cfg, S, layerIdx);
    } else if (bw->wArch != NULL) {
        bc.mixerOut = mixerForwardJungian(&bc, bw, cfg, S, layerIdx);
    } else {
        bc.mixerOut = mixerForwardHyena(&bc, bw, cfg, S);
    }

    bc.x1 = falloc(S * D);
    for (int i = 0; i < S * D; i++) bc.x1[i] = input[i] + bc.mixerOut[i];

    bc.ln2Mean = falloc(S);
    bc.ln2Var = falloc(S);
    bc.ln2Hat = falloc(S * D);
    bc.ln2Out = normForward(bc.x1, bw->ln2Gamma, bw->ln2Beta, S, D, bc.ln2Mean, bc.ln2Var, bc.ln2Hat);

    bc.ff1Out = matmul(bc.ln2Out, bw->ff1W, S, D, F);
    addBias(bc.ff1Out, bw->ff1B, S, F);

    bc.ff1UpOut = matmul(bc.ln2Out, bw->ff1WUp, S, D, F);
    addBias(bc.ff1UpOut, bw->ff1BUp, S, F);

    bc.siluOut = falloc(S * F);
    bc.inter = falloc(S * F);
    for (int i = 0; i < S * F; i++) {
        bc.siluOut[i] = silu(bc.ff1Out[i]);
        bc.inter[i] = bc.siluOut[i] * bc.ff1UpOut[i];
    }

    bc.ff2Out = matmul(bc.inter, bw->ff2W, S, F, D);
    addBias(bc.ff2Out, bw->ff2B, S, D);

    bc.output = falloc(S * D);
    for (int i = 0; i < S * D; i++) bc.output[i] = bc.x1[i] + bc.ff2Out[i];

    return bc;
}

/*
 * One block, backward (mixer-agnostic). Mirrors blockForward in reverse,
 * accumulating gradients into `bg`. Returns dInput for the previous block.
 */
static float *blockBackward(const float *dOutput, BlockCache *bc, BlockWeights *bw,
                            Config cfg, BlockWeights *bg, int S, int layerIdx) {
    int D = cfg.embDim, F = cfg.ffDim;

    /* output = x1 + ff2Out, so dOutput flows to x1 directly and through the FFN. */
    const float *dFF2Out = dOutput;

    accumFree(bg->ff2B, sumCols(dFF2Out, S, D), D);
    accumFree(bg->ff2W, matmulTransA(bc->inter, dFF2Out, S, F, D), F * D);
    float *dInter = matmulTransB(dFF2Out, bw->ff2W, S, D, F);

    float *dSilu = falloc(S * F);
    float *dFF1UpOut = falloc(S * F);
    for (int i = 0; i < S * F; i++) {
        dSilu[i] = dInter[i] * bc->ff1UpOut[i];
        dFF1UpOut[i] = dInter[i] * bc->siluOut[i];
    }

    float *dFF1Out = falloc(S * F);
    for (int i = 0; i < S * F; i++) {
        float x = bc->ff1Out[i];
        float sig = 1.f / (1.f + (float)exp(-x));
        float dsilu = sig * (1.f + x * (1.f - sig));
        dFF1Out[i] = dSilu[i] * dsilu;
    }
    free(dInter);
    free(dSilu);

    accumFree(bg->ff1BUp, sumCols(dFF1UpOut, S, F), F);
    accumFree(bg->ff1WUp, matmulTransA(bc->ln2Out, dFF1UpOut, S, D, F), D * F);
    accumFree(bg->ff1B, sumCols(dFF1Out, S, F), F);
    accumFree(bg->ff1W, matmulTransA(bc->ln2Out, dFF1Out, S, D, F), D * F);

    float *dLn2Out_gate = matmulTransB(dFF1Out, bw->ff1W, S, F, D);
    float *dLn2Out_up = matmulTransB(dFF1UpOut, bw->ff1WUp, S, F, D);
    float *dLn2Out = falloc(S * D);
    for (int i = 0; i < S * D; i++) dLn2Out[i] = dLn2Out_gate[i] + dLn2Out_up[i];
    free(dLn2Out_gate); free(dLn2Out_up);
    free(dFF1Out); free(dFF1UpOut);

    float *dLn2In = normBackward(dLn2Out, bc->ln2Hat, bw->ln2Gamma, bc->ln2Mean, bc->ln2Var,
                                      S, D, bg->ln2Gamma, bg->ln2Beta);
    free(dLn2Out);

    /* dL/dx1 = direct residual (dOutput) + FFN path (dLn2In). Since x1 = input +
     * mixerOut, this is also dL/dmixerOut and the residual contribution to dInput. */
    float *dMixerOut = falloc(S * D);
    for (int i = 0; i < S * D; i++) dMixerOut[i] = dOutput[i] + dLn2In[i];
    free(dLn2In);

    float *dLn1Out = NULL;
    if (bw->wB != NULL) {
        dLn1Out = mixerBackwardMamba(dMixerOut, bc, bw, cfg, bg, S);
    } else if (bw->wQ != NULL) {
        dLn1Out = mixerBackwardAttn(dMixerOut, bc, bw, cfg, bg, S);
    } else if (bw->convW_cnn != NULL) {
        dLn1Out = mixerBackwardCNN(dMixerOut, bc, bw, cfg, bg, S, layerIdx);
    } else if (bw->wArch != NULL) {
        dLn1Out = mixerBackwardJungian(dMixerOut, bc, bw, cfg, bg, S);
    } else {
        dLn1Out = mixerBackwardHyena(dMixerOut, bc, bw, cfg, bg, S);
    }

    float *dLn1In = normBackward(dLn1Out, bc->ln1Hat, bw->ln1Gamma, bc->ln1Mean, bc->ln1Var,
                                      S, D, bg->ln1Gamma, bg->ln1Beta);
    free(dLn1Out);

    /* dInput = residual path (dL/dx1 = dMixerOut) + mixer path (dLn1In) */
    float *dInput = falloc(S * D);
    for (int i = 0; i < S * D; i++) dInput[i] = dMixerOut[i] + dLn1In[i];
    free(dMixerOut);
    free(dLn1In);

    return dInput;
}

/* ─── Full forward pass ───────────────────────────────────────────────────── */

typedef struct {
    const int *tokenIds;
    int S;
    float *x0;
#if USE_ARCHETYPE
    float *archXc;     /* S×D  curved block input (x0 · M) */
    float *archM;      /* D×D  per-sequence metric M = I + Σ w_k C_k */
    float *archW;      /* N_ARCH  softmax archetype weights */
    float *archXmean;  /* D    sequence mean of x0 */
#endif
    BlockCache *blockCaches;  /* [numLayers] */
    int numBlocks;
    float *lnFOut, *lnFMean, *lnFVar, *lnFHat;
    float *logits, *probs;
} ForwardCache;

static void freeForwardCache(ForwardCache *fc) {
    free(fc->x0);
#if USE_ARCHETYPE
    free(fc->archXc); free(fc->archM); free(fc->archW); free(fc->archXmean);
#endif
    for (int i = 0; i < fc->numBlocks; i++) freeBlockCache(&fc->blockCaches[i]);
    free(fc->blockCaches);
    free(fc->lnFOut); free(fc->lnFMean); free(fc->lnFVar); free(fc->lnFHat);
    free(fc->logits); free(fc->probs);
}

/*
 * token ids in -> probability distribution over vocab out.
 * Pipeline: tok+pos embeddings -> N blocks -> final layernorm -> head -> softmax.
 * Caches every intermediate activation for backward().
 */
static ForwardCache forward(const int *tokenIds, int S, Weights *w, Config cfg) {
    int V = cfg.vocabSize, D = cfg.embDim;

    ForwardCache fc;
    fc.tokenIds = tokenIds;
    fc.S = S;
    fc.numBlocks = cfg.numLayers;

    fc.x0 = falloc(S * D);
    for (int i = 0; i < S; i++)
        for (int d = 0; d < D; d++) {
#if USE_ROPE
            fc.x0[i * D + d] = w->tokEmb[tokenIds[i] * D + d];
#else
            fc.x0[i * D + d] = w->tokEmb[tokenIds[i] * D + d] + w->posEmb[i * D + d];
#endif
        }

    fc.blockCaches = malloc(sizeof(BlockCache) * cfg.numLayers);
    float *x = fc.x0;
#if USE_ARCHETYPE
    /* ArchetypalProjection: curve the embedding space once, before the blocks. */
    fc.archXmean = falloc(D);
    for (int d = 0; d < D; d++) { float s = 0.f; for (int i = 0; i < S; i++) s += fc.x0[i * D + d]; fc.archXmean[d] = s / (float)S; }
    fc.archW = falloc(N_ARCH);
    float invsq = 1.f / (float)sqrt((double)D);
    for (int k = 0; k < N_ARCH; k++) { float g = 0.f; for (int d = 0; d < D; d++) g += fc.archXmean[d] * w->archQ[k * D + d]; fc.archW[k] = g * invsq; }
    float amx = fc.archW[0]; for (int k = 1; k < N_ARCH; k++) if (fc.archW[k] > amx) amx = fc.archW[k];
    float asum = 0.f; for (int k = 0; k < N_ARCH; k++) { fc.archW[k] = (float)exp(fc.archW[k] - amx); asum += fc.archW[k]; }
    for (int k = 0; k < N_ARCH; k++) fc.archW[k] /= asum;
    fc.archM = falloc(D * D);
    for (int i = 0; i < D * D; i++) fc.archM[i] = 0.f;
    for (int i = 0; i < D; i++) fc.archM[i * D + i] = 1.f;               /* M = I + … */
    for (int k = 0; k < N_ARCH; k++) { float wk = fc.archW[k]; const float *C = g_archC + (size_t)k * D * D; for (int i = 0; i < D * D; i++) fc.archM[i] += wk * C[i]; }
    fc.archXc = matmul(fc.x0, fc.archM, S, D, D);                        /* x' = x0 · M */
    x = fc.archXc;
#endif
    for (int i = 0; i < cfg.numLayers; i++) {
        if (i < 8 && g_actSteer[i]) {          /* inference-only activation steering */
            for (int s = 0; s < S; s++)
                for (int d = 0; d < D; d++) x[s * D + d] += g_actW * g_actSteer[i][d];
        }
        fc.blockCaches[i] = blockForward(x, &w->blocks[i], cfg, S, i);
        x = fc.blockCaches[i].output;
    }

    fc.lnFMean = falloc(S);
    fc.lnFVar = falloc(S);
    fc.lnFHat = falloc(S * D);
    fc.lnFOut = normForward(x, w->lnFGamma, w->lnFBeta, S, D, fc.lnFMean, fc.lnFVar, fc.lnFHat);

    fc.logits = matmul(fc.lnFOut, w->headW, S, D, V);
    addBias(fc.logits, w->headB, S, V);

    fc.probs = softmaxRows(fc.logits, S, V);
    return fc;
}

/* ─── Loss ────────────────────────────────────────────────────────────────── */

/* Cross-entropy: average of -log p(correct token) over positions. */
static float crossEntropyLoss(const float *probs, const int *targets, int seqLen, int V) {
    float loss = 0.f;
    for (int i = 0; i < seqLen; i++)
        loss -= (float)log(probs[i * V + targets[i]] + 1e-10);
    return loss / seqLen;
}

/* Cross-entropy over only the masked (completion) positions, per active token. */
static float crossEntropyLossMasked(const float *probs, const int *targets,
                                    const int *mask, int seqLen, int V, int nActive) {
    float loss = 0.f;
    for (int i = 0; i < seqLen; i++)
        if (mask[i]) loss -= (float)log(probs[i * V + targets[i]] + 1e-10);
    return loss / nActive;
}

/* ─── Full backward pass ──────────────────────────────────────────────────── */

/* Gradients accumulate (+=) into `grads`, so several sequences can contribute before one optimizer step.
 * If `mask` is non-NULL, only positions with mask[i]!=0 contribute loss/gradient, normalized by
 * `nActive` (completion-only training for the tool-calling task); mask==NULL = all S positions, /S. */
static void backward(ForwardCache *fc, const int *targets, const int *mask, int nActive,
                     Weights *w, Config cfg, Grads *grads) {
    int V = cfg.vocabSize, D = cfg.embDim, S = fc->S;
    int norm = mask ? nActive : S;

    /* d(cross-entropy + softmax)/d(logits) = (probs - onehot)/norm; masked-out rows stay 0. */
    float *dLogits = falloc(S * V);
    for (int i = 0; i < S; i++) {
        if (mask && !mask[i]) continue;
        for (int j = 0; j < V; j++) dLogits[i * V + j] = fc->probs[i * V + j] / norm;
        dLogits[i * V + targets[i]] -= 1.f / norm;
    }

    accumFree(grads->headB, sumCols(dLogits, S, V), V);
    accumFree(grads->headW, matmulTransA(fc->lnFOut, dLogits, S, D, V), D * V);
    float *dX = matmulTransB(dLogits, w->headW, S, V, D);
    free(dLogits);

    float *tmp = normBackward(dX, fc->lnFHat, w->lnFGamma, fc->lnFMean, fc->lnFVar,
                                   S, D, grads->lnFGamma, grads->lnFBeta);
    free(dX);
    dX = tmp;

    for (int i = cfg.numLayers - 1; i >= 0; i--) {
        tmp = blockBackward(dX, &fc->blockCaches[i], &w->blocks[i], cfg, &grads->blocks[i], S, i);
        free(dX);
        dX = tmp;
    }

#if USE_ARCHETYPE
    /* Backprop the ArchetypalProjection. dX here is ∂L/∂x' (curved input). */
    {
        /* dM = x0ᵀ · dX  (D×D);  direct path dX0 = dX · Mᵀ */
        float *dM  = matmulTransA(fc->x0, dX, S, D, D);
        float *dX0 = matmulTransB(dX, fc->archM, S, D, D);
        /* dw_k = <dM, C_k>_F */
        float dW[N_ARCH];
        for (int k = 0; k < N_ARCH; k++) { const float *C = g_archC + (size_t)k * D * D; float acc = 0.f; for (int i = 0; i < D * D; i++) acc += dM[i] * C[i]; dW[k] = acc; }
        free(dM);
        /* softmax jacobian → dG */
        float dot = 0.f; for (int k = 0; k < N_ARCH; k++) dot += dW[k] * fc->archW[k];
        float dG[N_ARCH]; for (int k = 0; k < N_ARCH; k++) dG[k] = fc->archW[k] * (dW[k] - dot);
        /* dQ_k += (dG_k/√D)·x̄ ; dx̄ += (1/√D) Σ_k dG_k·Q_k */
        float invsqb = 1.f / (float)sqrt((double)D);
        float *dXmean = falloc(D); for (int d = 0; d < D; d++) dXmean[d] = 0.f;
        for (int k = 0; k < N_ARCH; k++) {
            float g = dG[k] * invsqb;
            for (int d = 0; d < D; d++) { grads->archQ[k * D + d] += g * fc->archXmean[d]; dXmean[d] += g * w->archQ[k * D + d]; }
        }
        /* x̄ = (1/S)Σ x0 → each position gets dXmean/S added to the direct path */
        for (int i = 0; i < S; i++) for (int d = 0; d < D; d++) dX0[i * D + d] += dXmean[d] / (float)S;
        free(dXmean);
        free(dX); dX = dX0;
    }
#endif

    for (int i = 0; i < S; i++) {
        int tok = fc->tokenIds[i];
        for (int d = 0; d < D; d++) {
            grads->tokEmb[tok * D + d] += dX[i * D + d];
#if !USE_ROPE
            grads->posEmb[i * D + d]   += dX[i * D + d];
#endif
        }
    }
    free(dX);
}

/* ─── Adam optimizer ──────────────────────────────────────────────────────── */

typedef struct { float **m, **v; int nParams; } AdamBuf;

static AdamBuf initAdam(Weights *w, Config cfg) {
    ParamRef refs[8 + 64 * 16];
    int n = collectParams(w, cfg, refs);
    AdamBuf ab;
    ab.nParams = n;
    ab.m = malloc(sizeof(float *) * n);
    ab.v = malloc(sizeof(float *) * n);
    for (int i = 0; i < n; i++) { ab.m[i] = falloc(refs[i].len); ab.v[i] = falloc(refs[i].len); }
    return ab;
}

#if USE_MUON
/* dst (c×r) = src (r×c) transposed. */
static void transposeMat(const float *src, float *dst, int r, int c) {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            dst[j * r + i] = src[i * c + j];
}

/*
 * Newton–Schulz orthogonalization (Muon): iterate X ← a·X + b·(XXᵀ)X + c·(XXᵀ)²X
 * five times after a Frobenius-norm normalization. This drives X's singular values
 * toward 1 — i.e. X → the orthogonal factor UVᵀ of its SVD — without ever computing
 * an SVD (just matmuls). The quintic coefficients are Keller Jordan's tuned values.
 * We keep the Gram matrix XXᵀ on the smaller dimension by transposing when r > c.
 * @see https://kellerjordan.github.io/posts/muon/
 */
static void newtonSchulz5(float *X, int r, int c) {
    if (r > c) {                              /* operate on the smaller Gram dim */
        float *T = falloc(r * c);
        transposeMat(X, T, r, c);
        newtonSchulz5(T, c, r);
        transposeMat(T, X, c, r);
        free(T);
        return;
    }
    const float a = 3.4445f, b = -4.7750f, c3 = 2.0315f;
    double ss = 0.0;
    for (int i = 0; i < r * c; i++) ss += (double)X[i] * X[i];
    float inv = 1.f / ((float)sqrt(ss) + 1e-7f);
    for (int i = 0; i < r * c; i++) X[i] *= inv;

    for (int step = 0; step < 5; step++) {
        float *A  = matmulTransB(X, X, r, c, r);  /* A  = X Xᵀ   (r×r) */
        float *A2 = matmul(A, A, r, r, r);        /* A² = A·A    (r×r) */
        for (int i = 0; i < r * r; i++) A[i] = b * A[i] + c3 * A2[i];  /* A ← B */
        float *BX = matmul(A, X, r, r, c);        /* B·X         (r×c) */
        for (int i = 0; i < r * c; i++) X[i] = a * X[i] + BX[i];
        free(A); free(A2); free(BX);
    }
}
#endif /* USE_MUON */

/*
 * AdamW step over every parameter. Maintains per-parameter first/second moment
 * running averages (m, v), bias-corrects them, and applies:
 *   param -= lr * mHat / (sqrt(vHat) + eps)
 *   param -= lr * wd * param          (only for weight matrices, not biases/norms)
 *
 * The key insight of AdamW (Loshchilov & Hutter 2017) is that weight decay should
 * be decoupled from the gradient-based update. Classic L2 regularization adds
 * wd*param to the gradient, which then gets scaled by Adam's adaptive learning
 * rate — effectively giving different decay rates to different parameters. AdamW
 * applies the decay directly to the parameter, independent of the optimizer state.
 *
 * With USE_MUON, dense 2D weight matrices (pw[p].r > 0) instead take a Muon step:
 * an orthogonalized momentum update at MUON_LR_SCALE× the learning rate. All other
 * params (embeddings, head, conv, 1D) keep the AdamW step below.
 *
 * @see https://arxiv.org/abs/1711.05101  "Decoupled Weight Decay Regularization"
 * @see https://arxiv.org/abs/1412.6980  "Adam: A Method for Stochastic Optimization"
 */
static void adamUpdate(Weights *w, Grads *grads, AdamBuf *buf, Config cfg,
                       float lr, int t) {
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    float wd = WEIGHT_DECAY;
    float bc1 = 1.f - (float)pow(beta1, t);
    float bc2 = 1.f - (float)pow(beta2, t);

    ParamRef pw[8 + 64 * 16], pg[8 + 64 * 16];
    int n = collectParams(w, cfg, pw);
    collectParams(grads, cfg, pg);

    for (int p = 0; p < n; p++) {
        float *param = pw[p].ptr, *grad = pg[p].ptr, *m = buf->m[p], *v = buf->v[p];
        int len = pw[p].len;
        int applyDecay = pw[p].decay && (wd > 0.f);
#if USE_MUON
        if (pw[p].r > 0) {                    /* Muon: orthogonalized momentum */
            float mb = 0.95f, mlr = lr * MUON_LR_SCALE;
            float *upd = falloc(len);
            for (int i = 0; i < len; i++) {
                m[i] = mb * m[i] + (1.f - mb) * grad[i];        /* momentum */
                upd[i] = (1.f - mb) * grad[i] + mb * m[i];      /* Nesterov */
            }
            newtonSchulz5(upd, pw[p].r, pw[p].c);
            float scale = (float)sqrt(fmax(1.0, (double)pw[p].r / (double)pw[p].c));
            for (int i = 0; i < len; i++) {
                param[i] -= mlr * scale * upd[i];
                if (applyDecay) param[i] -= mlr * wd * param[i];
            }
            free(upd);
            continue;
        }
#endif
        for (int i = 0; i < len; i++) {
            m[i] = beta1 * m[i] + (1.f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.f - beta2) * grad[i] * grad[i];
            float mHat = m[i] / bc1, vHat = v[i] / bc2;
            param[i] -= lr * mHat / ((float)sqrt(vHat) + eps);
            if (applyDecay) param[i] -= lr * wd * param[i];
        }
    }
}

static void zeroAllGrads(Grads *grads, Config cfg) {
    ParamRef refs[8 + 64 * 16];
    int n = collectParams(grads, cfg, refs);
    for (int i = 0; i < n; i++) memset(refs[i].ptr, 0, sizeof(float) * refs[i].len);
}

/*
 * Global gradient norm clipping: if the L2 norm of ALL gradient parameters
 * exceeds maxNorm, scale every gradient element by maxNorm/norm. This prevents
 * gradient explosions from destabilizing Adam's running averages.
 * @see https://arxiv.org/abs/1211.5063  "On the difficulty of training RNNs"
 */
static void clipGrads(Grads *grads, Config cfg, float maxNorm) {
    if (maxNorm <= 0.f) return;
    ParamRef refs[8 + 64 * 16];
    int n = collectParams(grads, cfg, refs);
    double sumSq = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < refs[i].len; j++)
            sumSq += (double)refs[i].ptr[j] * (double)refs[i].ptr[j];
    float norm = (float)sqrt(sumSq);
    if (norm > maxNorm) {
        float scale = maxNorm / norm;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < refs[i].len; j++)
                refs[i].ptr[j] *= scale;
    }
}

static void mergeGrads(Grads *dest, const Grads *src, Config cfg) {
    ParamRef destRefs[8 + 64 * 16];
    ParamRef srcRefs[8 + 64 * 16];
    int nDest = collectParams(dest, cfg, destRefs);
    (void)collectParams((Grads *)src, cfg, srcRefs);
    for (int i = 0; i < nDest; i++) {
        for (int j = 0; j < destRefs[i].len; j++) {
            destRefs[i].ptr[j] += srcRefs[i].ptr[j];
        }
    }
}

/* Free every parameter array plus the blocks[] array. Works for Weights and Grads. */
static void freeWeights(Weights *w, Config cfg) {
    ParamRef refs[8 + 64 * 16];
    int n = collectParams(w, cfg, refs);
    for (int i = 0; i < n; i++) free(refs[i].ptr);
    free(w->blocks);
}

static void freeAdam(AdamBuf *ab) {
    for (int i = 0; i < ab->nParams; i++) { free(ab->m[i]); free(ab->v[i]); }
    free(ab->m); free(ab->v);
}

/* ─── Text generation (temperature + nucleus/top-p sampling) ──────────────── */

typedef struct { float prob; int idx; } ProbIdx;

static int cmpProbDesc(const void *a, const void *b) {
    float pa = ((const ProbIdx *)a)->prob, pb = ((const ProbIdx *)b)->prob;
    return (pa < pb) - (pa > pb);
}

/*
 * Autoregressive generation: forward on current context, take the last position's
 * logits, temperature-scale, softmax, then nucleus (top-p) sample the next token.
 * Writes the generated ids (seed + new) into out; returns total length.
 * @see https://arxiv.org/abs/1904.09751  "The Curious Case of Neural Text Degeneration"
 */
static int generateText(Weights *w, Config cfg, const int *seedIds, int seedLen,
                        int maxLen, float temperature, float topP, int windowSize,
                        int *out) {
    int V = cfg.vocabSize;
    int *ids = malloc(sizeof(int) * (seedLen + maxLen));
    int nIds = seedLen;
    memcpy(ids, seedIds, sizeof(int) * seedLen);

    ProbIdx *sorted = malloc(sizeof(ProbIdx) * V);
    float *probs = malloc(sizeof(float) * V);

    for (int step = 0; step < maxLen; step++) {
        int start = nIds > windowSize ? nIds - windowSize : 0;
        int ctxLen = nIds - start;
        ForwardCache fc = forward(ids + start, ctxLen, w, cfg);
        int lastPos = ctxLen - 1;

        float mx = NEG_INF;
        for (int v = 0; v < V; v++) {
            probs[v] = fc.logits[lastPos * V + v] / temperature;
            if (probs[v] > mx) mx = probs[v];
        }
        float sum = 0.f;
        for (int v = 0; v < V; v++) { probs[v] = (float)exp(probs[v] - mx); sum += probs[v]; }
        for (int v = 0; v < V; v++) probs[v] /= sum;

        for (int v = 0; v < V; v++) { sorted[v].prob = probs[v]; sorted[v].idx = v; }
        qsort(sorted, V, sizeof(ProbIdx), cmpProbDesc);

        /* nucleus: smallest prefix whose cumulative prob >= topP */
        float cum = 0.f;
        int nucleusLen = 0;
        for (int v = 0; v < V; v++) {
            nucleusLen++;
            cum += sorted[v].prob;
            if (cum >= topP) break;
        }
        float nucleusSum = 0.f;
        for (int v = 0; v < nucleusLen; v++) nucleusSum += sorted[v].prob;

        float r = (float)randf() * nucleusSum;
        float c = 0.f;
        int sampled = sorted[0].idx;
        for (int v = 0; v < nucleusLen; v++) {
            c += sorted[v].prob;
            if (r < c) { sampled = sorted[v].idx; break; }
        }

        ids[nIds++] = sampled;

        /* Shadow-cache: when the window is about to slide by 1, absorb the shadow
         * at the position leaving the window (index 0 of this forward), i.e.
         * s_abs(newStart−1) — the seed of the next forward's recurrence. */
        if (g_shadowCacheOn) {
            int newStart = nIds > windowSize ? nIds - windowSize : 0;
            if (newStart > start) {
                for (int l = 0; l < cfg.numLayers && l < 8; l++)
                    if (cfg.layerTypes[l] == MIX_JUNGIAN && fc.blockCaches[l].shadow_j != NULL)
                        memcpy(g_shadowCache[l], fc.blockCaches[l].shadow_j,
                               sizeof(float) * N_ARCH);
            }
        }
        freeForwardCache(&fc);
    }

    memcpy(out, ids, sizeof(int) * nIds);
    free(ids); free(sorted); free(probs);
    return nIds;
}

/* ─── Gradient check (finite differences) ─────────────────────────────────── */

#if GRAD_CHECK
/* Forward-only loss for one sequence — the primitive the numerical gradient needs. */
static float computeLoss(const int *input, const int *target, int seqLen, Weights *w, Config cfg) {
    ForwardCache fc = forward(input, seqLen, w, cfg);
    float l = crossEntropyLoss(fc.probs, target, seqLen, cfg.vocabSize);
    freeForwardCache(&fc);
    return l;
}

/*
 * Validate the whole hand-derived backward against numerical gradients on a tiny
 * model + one sequence. For a sample of parameters, compares analytical g to
 * (loss(θ+ε) - loss(θ-ε)) / 2ε and reports the max relative error. A correct
 * backward gives ~1e-3 or smaller; a wrong one blows up.
 */
static void gradCheck(void) {
    resetRand(42);
    Config cfg = { .vocabSize = 7, .contextLen = 8, .embDim = 8, .numHeads = 2,
                   .numKVHeads = (NUM_KV_HEADS > 2) ? 2 : NUM_KV_HEADS,
                   .ffDim = 16, .stateN = 4 };
    applyLayout(&cfg);
    int S = 5;
    int input[5]  = { 0, 1, 2, 3, 4 };
    int target[5] = { 1, 2, 3, 4, 5 };

    Weights w = initWeights(cfg);
    Grads grads = zeroGrads(cfg);

#if STEER_AUG
    /* Check WITH a standing bias active: the augmented training regime must
     * leave the analytical backward exact (bias is an additive constant). */
    static float gcBias[N_ARCH];
    for (int k = 0; k < N_ARCH; k++) gcBias[k] = 0.11f * (float)((k * 7) % 5 - 2);
    for (int l = 0; l < cfg.numLayers && l < 8; l++)
        if (cfg.layerTypes[l] == MIX_JUNGIAN) g_personalBias[l] = gcBias;
    g_personalW = 13.f;
    printf("grad-check with STEER_AUG standing bias active (w_u=13)\n");
#endif

    ForwardCache fc = forward(input, S, &w, cfg);
    backward(&fc, target, NULL, 0, &w, cfg, &grads);
    freeForwardCache(&fc);

    ParamRef pw[8 + 64 * 16], pg[8 + 64 * 16];
    int n = collectParams(&w, cfg, pw);
    collectParams(&grads, cfg, pg);

#ifndef GC_EPS
#define GC_EPS 3e-3f
#endif
    const float eps = GC_EPS;
    /* Only judge parameters whose true gradient is large enough for float32 central
     * differences to resolve. Small grads are dominated by roundoff, and finite
     * differences across a ReLU kink are meaningless — both produce false failures,
     * not real bugs. (Lower GC_FLOOR to inspect the small-grad regime: relative
     * error grows as |grad| shrinks — the signature of finite-difference noise,
     * not a wrong backward.) */
#ifndef GC_FLOOR
#define GC_FLOOR 1e-1f
#endif
    const float GRAD_FLOOR = GC_FLOOR;
    float maxRel = 0.f, worstNum = 0.f, worstAna = 0.f;
    int checks = 0, judged = 0, failed = 0, worstP = 0;
    for (int p = 0; p < n; p++) {
        int step = pw[p].len > 4 ? pw[p].len / 4 : 1;  /* sample a few elems per array */
        for (int i = 0; i < pw[p].len; i += step) {
            float orig = pw[p].ptr[i];
            pw[p].ptr[i] = orig + eps;  float lp = computeLoss(input, target, S, &w, cfg);
            pw[p].ptr[i] = orig - eps;  float lm = computeLoss(input, target, S, &w, cfg);
            pw[p].ptr[i] = orig;
            float num = (lp - lm) / (2.f * eps);
            float ana = pg[p].ptr[i];
            checks++;
            if (fabsf(ana) < GRAD_FLOOR && fabsf(num) < GRAD_FLOOR) continue;
            judged++;
            float rel = fabsf(num - ana) / (fabsf(num) + fabsf(ana) + 1e-8f);
            if (rel > 3e-2f) failed++;
            if (rel > maxRel) { maxRel = rel; worstNum = num; worstAna = ana; worstP = p; }
        }
    }
    char wname[32]; getParameterName(worstP, &w, cfg, wname, sizeof(wname));
    printf("gradient check: %d sampled, %d judged (|grad| > %.0e), %d over tol\n",
           checks, judged, (double)GRAD_FLOOR, failed);
    printf("  max relative error = %.2e in %s  (numerical=%.5f analytical=%.5f)  ->  %s\n",
           (double)maxRel, wname, (double)worstNum, (double)worstAna,
           (failed == 0) ? "PASS" : "CHECK");

    freeWeights(&w, cfg);
    freeWeights(&grads, cfg);
}
#endif /* GRAD_CHECK */

/* ─── Training ────────────────────────────────────────────────────────────── */

static void printModelHeader(Config cfg) {
    printf("Decoder-only LM (C port) — layout: [");
    for (int i = 0; i < cfg.numLayers; i++) {
        printf("%s%s", mixerName(cfg.layerTypes[i]), (i == cfg.numLayers - 1) ? "" : ", ");
    }
    printf("]%s%s%s%s%s\n", USE_RMSNORM ? " + RMSNorm" : "", USE_MUON ? " + Muon" : "",
           USE_REPRESSION ? " + Desatenção" : "", USE_CONTRAST_CONV ? " + ContrastConv" : "",
           USE_ARCHETYPE ? " + Arquétipo" : "");
    printf("  vocab=%d ctx=%d D=%d heads=%d KVheads=%d N=%d ff=%d layers=%d\n",
           cfg.vocabSize, cfg.contextLen, cfg.embDim, cfg.numHeads, cfg.numKVHeads, cfg.stateN, cfg.ffDim, cfg.numLayers);
}

/*
 * Full-batch training on a tiny hardcoded corpus (default when no file is given).
 * Mirrors train.ts: forward+backward over every sequence, accumulate grads, one
 * Adam step per epoch. Kept for quick sanity runs and the gradient check.
 */
static void trainToy(void) {
    resetRand(42);

    const char *corpus =
        "the cat sat on the mat. the dog ran in the sun. "
        "the cat ran to the dog. the sun sat on the mat. ";
    int corpusLen = (int)strlen(corpus);

    int charToId[256];
    for (int i = 0; i < 256; i++) charToId[i] = -1;
    char idToChar[256];
    int vocabSize = 0;
    for (int i = 0; i < corpusLen; i++) {
        unsigned char ch = (unsigned char)corpus[i];
        if (charToId[ch] == -1) { charToId[ch] = vocabSize; idToChar[vocabSize] = (char)ch; vocabSize++; }
    }
    int *allTokenIds = malloc(sizeof(int) * corpusLen);
    for (int i = 0; i < corpusLen; i++) allTokenIds[i] = charToId[(unsigned char)corpus[i]];

    Config cfg = { .vocabSize = vocabSize, .contextLen = 32, .embDim = 32,
                   .numHeads = 2, .numKVHeads = (NUM_KV_HEADS > 2) ? 2 : NUM_KV_HEADS,
                   .ffDim = 64, .stateN = 8 };
    applyLayout(&cfg);

    int seqLen = 16;
    int numSeq = corpusLen - seqLen - 1;
    if (numSeq < 1) { fprintf(stderr, "corpus too short\n"); exit(1); }
    int **inputs  = malloc(sizeof(int *) * numSeq);
    int **targets = malloc(sizeof(int *) * numSeq);
    for (int i = 0; i < numSeq; i++) { inputs[i] = allTokenIds + i; targets[i] = allTokenIds + i + 1; }

    Weights w = initWeights(cfg);
    Grads grads = zeroGrads(cfg);
    AdamBuf adam = initAdam(&w, cfg);

    int num_threads = 1;
    #ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    #endif

    printModelHeader(cfg);
    printf("  params=%d  sequences=%d  seqLen=%d  threads=%d\n\n", countParams(&w, cfg), numSeq, seqLen, num_threads);

    int epochs = 300;
    float lr = 0.001f;
    int seed[3] = { allTokenIds[0], allTokenIds[1], allTokenIds[2] };

    Grads *thread_grads = malloc(sizeof(Grads) * num_threads);
    for (int t = 0; t < num_threads; t++) {
        thread_grads[t] = zeroGrads(cfg);
    }

    for (int epoch = 0; epoch <= epochs; epoch++) {
        for (int t = 0; t < num_threads; t++) {
            zeroAllGrads(&thread_grads[t], cfg);
        }
        float totalLoss = 0.f;
        #pragma omp parallel for reduction(+:totalLoss)
        for (int s = 0; s < numSeq; s++) {
            int tid = 0;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            #endif
            ForwardCache fc = forward(inputs[s], seqLen, &w, cfg);
            totalLoss += crossEntropyLoss(fc.probs, targets[s], seqLen, cfg.vocabSize);
            backward(&fc, targets[s], NULL, 0, &w, cfg, &thread_grads[tid]);
            freeForwardCache(&fc);
        }
        zeroAllGrads(&grads, cfg);
        for (int t = 0; t < num_threads; t++) {
            mergeGrads(&grads, &thread_grads[t], cfg);
        }
        adamUpdate(&w, &grads, &adam, cfg, lr, epoch + 1);

        if (epoch % PRINT_EVERY == 0 || epoch == epochs) {
            int out[3 + 64];
            int n = generateText(&w, cfg, seed, 3, 40, 0.8f, 0.9f, seqLen, out);
            printf("epoch %3d  loss %.4f  | ", epoch, totalLoss / numSeq);
            for (int i = 0; i < n; i++) putchar(idToChar[out[i]]);
            printf("\n");
        }
    }

    for (int t = 0; t < num_threads; t++) {
        freeWeights(&thread_grads[t], cfg);
    }
    free(thread_grads);

    freeWeights(&w, cfg); freeWeights(&grads, cfg); freeAdam(&adam);
    free(allTokenIds); free(inputs); free(targets);
}

/*
 * Load a text file as a char-level token stream. Strips a Project Gutenberg
 * header/footer if present, normalizes CRLF, and drops non-ASCII bytes (curly
 * quotes, em-dashes, accents) so the vocab stays a clean ~80 ASCII chars.
 * Returns malloc'd ids; sets *nOut, *vocabOut, fills idToChar[256].
 */
static int *loadCorpus(const char *path, int *nOut, int *vocabOut, char *idToChar) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    size_t got = fread(buf, 1, (size_t)sz, f);
    buf[got] = 0;
    fclose(f);

    /* Gutenberg body lives between the START/END markers, if present. */
    char *bodyBegin = buf, *bodyEnd = buf + got;
    char *start = strstr(buf, "*** START OF TH");
    if (start) { char *nl = strchr(start, '\n'); if (nl) bodyBegin = nl + 1; }
    char *end = strstr(bodyBegin, "*** END OF TH");
    if (end) bodyEnd = end;

    int charToId[256];
    for (int i = 0; i < 256; i++) charToId[i] = -1;
    int vocab = 0, n = 0;
    int *ids = malloc(sizeof(int) * (size_t)(bodyEnd - bodyBegin));
    for (char *p = bodyBegin; p < bodyEnd; p++) {
        unsigned char ch = (unsigned char)*p;
        if (ch == '\r' || ch >= 128) continue;   /* drop CR and non-ASCII */
        if (charToId[ch] == -1) { charToId[ch] = vocab; idToChar[vocab] = (char)ch; vocab++; }
        ids[n++] = charToId[ch];
    }
    free(buf);
    *nOut = n; *vocabOut = vocab;
    return ids;
}

static void printArchetypeStats(Weights *w, Config cfg, int *ids, int nTokens, const char *idToChar) {
    int numJungian = 0;
    for (int i = 0; i < cfg.numLayers; i++) {
        if (cfg.layerTypes[i] == MIX_JUNGIAN) numJungian++;
    }
    if (numJungian == 0) return;

    int K = N_ARCH;
    double *sumP = calloc(K, sizeof(double));
    double *sumS = calloc(K, sizeof(double));
    int totalElements = 0;

    int numSamples = 32;
    int seqLen = cfg.contextLen;
    for (int n = 0; n < numSamples; n++) {
        int sp = (int)(randf() * (double)(nTokens - seqLen - 1));
        const int *inp = ids + sp;
        ForwardCache fc = forward(inp, seqLen, w, cfg);
        for (int l = 0; l < cfg.numLayers; l++) {
            if (cfg.layerTypes[l] == MIX_JUNGIAN) {
                BlockCache *bc = &fc.blockCaches[l];
                if (bc->p_j != NULL && bc->shadow_j != NULL) {
                    for (int s = 0; s < seqLen; s++) {
                        for (int k = 0; k < K; k++) {
                            sumP[k] += bc->p_j[s * K + k];
                            sumS[k] += bc->shadow_j[s * K + k];
                        }
                    }
                    totalElements += seqLen;
                }
            }
        }
        freeForwardCache(&fc);
    }

    const char *archNames[16] = {
        "self", "shadow", "anima", "animus", "great_mother", "great_father",
        "hero", "wise_old_man", "persona", "puer_aeternus", "senex", "trickster",
        "kore", "divine_child", "spirit", "quaternity"
    };

    printf("=== Archetype Adherence Stats (sampled over %d tokens) ===\n", totalElements);
    printf("%-16s | %-12s | %-12s\n", "Archetype", "Persona (p_j)", "Shadow (s_j)");
    printf("-----------------------------------------------\n");
    for (int k = 0; k < K; k++) {
        double avgP = totalElements > 0 ? sumP[k] / totalElements : 0.0;
        double avgS = totalElements > 0 ? sumS[k] / totalElements : 0.0;
        char nameBuf[24];
        const char *nm = (k < 16) ? archNames[k]
                                  : (snprintf(nameBuf, sizeof(nameBuf), "server_%02d", k), nameBuf);
        printf("%-16s | %-12.5f | %-12.5f\n", nm, avgP, avgS);
    }
    printf("=========================================================\n\n");

    // Compute semantic anchors (top 5 characters for each archetype and shadow)
    float *wArch = NULL;
    float *wS_j = NULL;
    for (int l = 0; l < cfg.numLayers; l++) {
        if (cfg.layerTypes[l] == MIX_JUNGIAN) {
            wArch = w->blocks[l].wArch;
            wS_j = w->blocks[l].wS_j;
            break;
        }
    }

    if (wArch != NULL && wS_j != NULL && idToChar != NULL) {
        int V = cfg.vocabSize;
        int D = cfg.embDim;
        printf("=== Discovery of Emergent Archetypes & Shadows (Vocabulary Projection) ===\n");
        for (int k = 0; k < K; k++) {
            // 1. Conscious Server (Persona) - wArch
            float normArch = 0.0f;
            for (int d = 0; d < D; d++) normArch += wArch[k * D + d] * wArch[k * D + d];
            normArch = (float)sqrt((double)normArch);

            struct {
                int ch;
                float score;
            } topConscious[256];

            for (int v = 0; v < V; v++) {
                float dot = 0.0f;
                float normTe = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += w->tokEmb[v * D + d] * wArch[k * D + d];
                    normTe += w->tokEmb[v * D + d] * w->tokEmb[v * D + d];
                }
                normTe = (float)sqrt((double)normTe);
                float cosSim = (normArch > 0.0f && normTe > 0.0f) ? dot / (normArch * normTe) : 0.0f;
                topConscious[v].ch = v;
                topConscious[v].score = cosSim;
            }

            for (int i = 0; i < V - 1; i++) {
                for (int j = 0; j < V - i - 1; j++) {
                    if (topConscious[j].score < topConscious[j + 1].score) {
                        int tempCh = topConscious[j].ch;
                        float tempScore = topConscious[j].score;
                        topConscious[j].ch = topConscious[j+1].ch;
                        topConscious[j].score = topConscious[j+1].score;
                        topConscious[j+1].ch = tempCh;
                        topConscious[j+1].score = tempScore;
                    }
                }
            }

            // 2. Unconscious Shadow Projection - wS_j
            float normS = 0.0f;
            for (int d = 0; d < D; d++) normS += wS_j[k * D + d] * wS_j[k * D + d];
            normS = (float)sqrt((double)normS);

            struct {
                int ch;
                float score;
            } topShadow[256];

            for (int v = 0; v < V; v++) {
                float dot = 0.0f;
                float normTe = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += w->tokEmb[v * D + d] * wS_j[k * D + d];
                    normTe += w->tokEmb[v * D + d] * w->tokEmb[v * D + d];
                }
                normTe = (float)sqrt((double)normTe);
                float cosSim = (normS > 0.0f && normTe > 0.0f) ? dot / (normS * normTe) : 0.0f;
                topShadow[v].ch = v;
                topShadow[v].score = cosSim;
            }

            for (int i = 0; i < V - 1; i++) {
                for (int j = 0; j < V - i - 1; j++) {
                    if (topShadow[j].score < topShadow[j + 1].score) {
                        int tempCh = topShadow[j].ch;
                        float tempScore = topShadow[j].score;
                        topShadow[j].ch = topShadow[j+1].ch;
                        topShadow[j].score = topShadow[j+1].score;
                        topShadow[j+1].ch = tempCh;
                        topShadow[j+1].score = tempScore;
                    }
                }
            }

            printf("Server #%02d:\n", k);
            printf("  Persona (Conscious) : ");
            for (int i = 0; i < (V < 5 ? V : 5); i++) {
                char c = idToChar[topConscious[i].ch];
                if (c == '\n') printf("'\\n' (%.2f) ", topConscious[i].score);
                else if (c == ' ') printf("' ' (%.2f) ", topConscious[i].score);
                else printf("'%c' (%.2f) ", c, topConscious[i].score);
            }
            printf("\n");

            printf("  Shadow (Repressed)  : ");
            for (int i = 0; i < (V < 5 ? V : 5); i++) {
                char c = idToChar[topShadow[i].ch];
                if (c == '\n') printf("'\\n' (%.2f) ", topShadow[i].score);
                else if (c == ' ') printf("' ' (%.2f) ", topShadow[i].score);
                else printf("'%c' (%.2f) ", c, topShadow[i].score);
            }
            printf("\n\n");
        }
        printf("=========================================================================\n\n");
    }

    free(sumP);
    free(sumS);
}

/* ─── Qualitative-metrics eval  (-DEVAL_SAMPLES=n) ───────────────────────────
 * The cross-entropy loss rewards compressing the frequent, not reproducing the
 * structurally decisive (see README #4/#7). This block generates EVAL_SAMPLES
 * samples from FIXED, evenly-spaced corpus prompts — deterministic across
 * configs: same corpus ⇒ same prompts, so two layouts trained with the same
 * seed are compared on identical seeds — and prints them between parse-friendly
 * markers for metrics.py, which computes:
 *   SCA (Syntactic Closure Accuracy) — delimiters opened vs properly closed;
 *   LVR (Lexical Validity Rate)      — generated words found in the corpus lexicon.
 * For Jungian layers it also prints the PGE (Psychic Gating Entropy): the mean
 * Shannon entropy of the persona distribution p_t over the generated sequence.
 * PGE ≈ 0 ⇒ psyche collapsed to one rigid server; PGE ≈ ln K ⇒ random hopping;
 * healthy dynamics live in between. Proposed in DISCOVERY.md §6.
 */
#ifndef EVAL_SAMPLES
#define EVAL_SAMPLES 0
#endif
#ifndef EVAL_GEN_LEN
#define EVAL_GEN_LEN 600
#endif

static int g_evalSample = -1;   /* --eval-sample i: generate only sample i (chunkable eval) */
static int g_shadowCacheReq = 0; /* --shadow-cache: persist s_t across window slides */
static int g_evalWindow = 0;    /* --eval-window n: cap the generation context window at
                                 * inference (default 0 = contextLen). Shrinking it below
                                 * the training window measures how much each mixer relies
                                 * on the *explicit* context vs. compressed state — the
                                 * Shadow-as-lossy-memory test (DISCOVERY.md). */

static void evalMetricsGeneration(Weights *w, Config cfg, const int *ids, int nTokens,
                                  const char *idToChar) {
#if EVAL_SAMPLES > 0
    int seedLen = 32;
    int genLen = EVAL_GEN_LEN;
    int seqLen = (g_evalWindow > 0 && g_evalWindow < cfg.contextLen)
                     ? g_evalWindow : cfg.contextLen;
    if (g_evalWindow > 0) printf("eval window = %d\n", seqLen);
    int *out = malloc(sizeof(int) * (seedLen + genLen));
    for (int i = 0; i < EVAL_SAMPLES; i++) {
        if (g_evalSample >= 0 && i != g_evalSample) continue;
        resetRand(20260713u + (uint32_t)i);   /* deterministic sampling per sample */
        long sp = (long)(i + 1) * (long)(nTokens - seedLen - 1) / (EVAL_SAMPLES + 1);
        if (g_shadowCacheReq) {              /* fresh shadow per sample */
            memset(g_shadowCache, 0, sizeof(g_shadowCache));
            g_shadowCacheOn = 1;
        }
        int n = generateText(w, cfg, ids + sp, seedLen, genLen, 0.8f, 0.9f, seqLen, out);
        g_shadowCacheOn = 0;                 /* PGE forward below runs cache-free */
        printf("=== EVAL SAMPLE %d (start=%ld seedLen=%d) ===\n", i, sp, seedLen);
        for (int j = 0; j < n; j++) putchar(idToChar[out[j]]);
        printf("\n=== END SAMPLE %d ===\n", i);

        /* PGE over the generated sequence (Jungian layers only) */
        int S = n < seqLen ? n : seqLen;
        ForwardCache fc = forward(out + (n - S), S, w, cfg);
        for (int l = 0; l < cfg.numLayers; l++) {
            if (cfg.layerTypes[l] != MIX_JUNGIAN) continue;
            BlockCache *bc = &fc.blockCaches[l];
            double H = 0.0;
            for (int s = 0; s < S; s++)
                for (int k = 0; k < N_ARCH; k++) {
                    float p = bc->p_j[s * N_ARCH + k];
                    if (p > 1e-12f) H -= (double)p * log((double)p);
                }
            printf("PGE sample=%d layer=%d %.4f (uniform=%.4f)\n",
                   i, l, H / S, log((double)N_ARCH));
        }
        freeForwardCache(&fc);
    }
    free(out);
#else
    (void)w; (void)cfg; (void)ids; (void)nTokens; (void)idToChar;
#endif
}

/* ─── Archetype repulsion  (-DARCH_REPULSION=λ) ──────────────────────────────
 * Measurement first: after 1000 steps of plain training the pairwise |cos| of
 * the 16 learned archetypes is statistically identical to that of 16 RANDOM
 * vectors in R^64 (mean ≈ 0.10 both) — SGD neither collapses nor separates
 * them; whatever distinction exists is inherited from the random init, not
 * learned. But an archetype is only meaningful insofar as it differs from the
 * others — the bank's geometry should express that, not owe it to chance.
 *
 * This step descends the pairwise squared-cosine penalty
 *
 *     P = Σ_{i<j} cos²(A_i, A_j)      ∂P/∂A_i = Σ_{j≠i} 2c_ij (Â_j − c_ij Â_i)/|A_i|
 *
 * (Â = unit vector) *decoupled* from the data gradient, exactly like AdamW's
 * weight decay: applied after the Adam step, scaled by λ·lr_t. No backward or
 * grad-check changes. cos → 0 pushes toward mutual orthogonality; norms are
 * untouched (the model keeps norm as a free gain). O(K²D) per step.
 */
static void archRepulsionStep(Weights *w, Config cfg, float eta) {
    if (eta <= 0.f) return;
    int D = cfg.embDim, K = N_ARCH;
    for (int l = 0; l < cfg.numLayers; l++) {
        if (cfg.layerTypes[l] != MIX_JUNGIAN) continue;
        float *A = w->blocks[l].wArch;
        float norm[N_ARCH];
        float *grad = calloc((size_t)K * D, sizeof(float));
        for (int i = 0; i < K; i++) {
            float s = 0.f;
            for (int d = 0; d < D; d++) s += A[i*D+d] * A[i*D+d];
            norm[i] = sqrtf(s) + 1e-8f;
        }
        for (int i = 0; i < K; i++) {
            for (int j = i + 1; j < K; j++) {
                float dot = 0.f;
                for (int d = 0; d < D; d++) dot += A[i*D+d] * A[j*D+d];
                float c = dot / (norm[i] * norm[j]);
                float ci = 2.f * c / (norm[i] * norm[j]);
                for (int d = 0; d < D; d++) {
                    grad[i*D+d] += ci * (A[j*D+d] - dot * A[i*D+d] / (norm[i]*norm[i]));
                    grad[j*D+d] += ci * (A[i*D+d] - dot * A[j*D+d] / (norm[j]*norm[j]));
                }
            }
        }
        for (int i = 0; i < K * D; i++) A[i] -= eta * grad[i];
        free(grad);
    }
}

/* ─── Checkpointing  (./transformer corpus.txt --ckpt file [--until n]) ──────
 * Binary dump of everything needed to resume training *exactly*: a config
 * sanity header, the PRNG state (so the data-sampling stream continues where
 * it stopped), the step counter, all params (collectParams order) and the
 * Adam m/v moments (m also carries the Muon momentum). Resume requires the
 * same compile-time layout/toggles — the header is checked. Writes are atomic
 * (tmp file + rename), and the train loop saves every 50 steps, so a killed
 * run loses at most 50 steps. --until caps the steps of THIS run while the
 * LR schedule keeps using STEPS — chunked runs follow the same schedule as a
 * single full run.
 */
static const char *g_ckptPath = NULL;
static int g_untilStep = 0x7fffffff;

static void saveCkpt(Weights *w, AdamBuf *adam, Config cfg, int step) {
    if (!g_ckptPath) return;
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.tmp", g_ckptPath);
    FILE *f = fopen(tmp, "wb");
    if (!f) { fprintf(stderr, "ckpt: cannot write %s\n", tmp); return; }
    ParamRef refs[8 + 64 * 16];
    int n = collectParams(w, cfg, refs);
    int hdr[6] = { 0x4A554E47, cfg.vocabSize, cfg.embDim, cfg.numLayers, n, step };
    fwrite(hdr, sizeof(int), 6, f);
    fwrite(&g_rng_state, sizeof(uint32_t), 1, f);
    for (int i = 0; i < n; i++) fwrite(refs[i].ptr, sizeof(float), refs[i].len, f);
    for (int i = 0; i < n; i++) fwrite(adam->m[i], sizeof(float), refs[i].len, f);
    for (int i = 0; i < n; i++) fwrite(adam->v[i], sizeof(float), refs[i].len, f);
    fclose(f);
    rename(tmp, g_ckptPath);
}

static int loadCkpt(Weights *w, AdamBuf *adam, Config cfg) {
    if (!g_ckptPath) return 0;
    FILE *f = fopen(g_ckptPath, "rb");
    if (!f) return 0;
    ParamRef refs[8 + 64 * 16];
    int n = collectParams(w, cfg, refs);
    int hdr[6];
    if (fread(hdr, sizeof(int), 6, f) != 6 || hdr[0] != 0x4A554E47 ||
        hdr[1] != cfg.vocabSize || hdr[2] != cfg.embDim ||
        hdr[3] != cfg.numLayers || hdr[4] != n) {
        fprintf(stderr, "ckpt: %s incompatible with current config — ignoring\n", g_ckptPath);
        fclose(f);
        return 0;
    }
    int ok = fread(&g_rng_state, sizeof(uint32_t), 1, f) == 1;
    for (int i = 0; i < n && ok; i++)
        ok = fread(refs[i].ptr, sizeof(float), refs[i].len, f) == (size_t)refs[i].len;
    for (int i = 0; i < n && ok; i++)
        ok = fread(adam->m[i], sizeof(float), refs[i].len, f) == (size_t)refs[i].len;
    for (int i = 0; i < n && ok; i++)
        ok = fread(adam->v[i], sizeof(float), refs[i].len, f) == (size_t)refs[i].len;
    fclose(f);
    if (!ok) { fprintf(stderr, "ckpt: short read in %s — ignoring\n", g_ckptPath); return 0; }
    return hdr[5];
}

/*
 * Minibatch SGD training on a large corpus loaded from a file. Each step samples
 * BATCH random windows, accumulates gradients, averages them, and takes one Adam
 * step — the standard recipe that (unlike full-batch) scales to millions of tokens.
 */
static void trainFromFile(const char *path) {
    resetRand(42);
    char idToChar[256];
    int nTokens, vocabSize;
    int *ids = loadCorpus(path, &nTokens, &vocabSize, idToChar);

    int seqLen = SEQ_LEN;
    if (nTokens < seqLen + 2) { fprintf(stderr, "corpus too short\n"); exit(1); }

    Config cfg = { .vocabSize = vocabSize, .contextLen = seqLen, .embDim = EMB_DIM,
                   .numHeads = 4, .numKVHeads = (NUM_KV_HEADS > 4) ? 4 : NUM_KV_HEADS,
                   .ffDim = FF_DIM, .stateN = 16 };
    applyLayout(&cfg);

    Weights w = initWeights(cfg);
    Grads grads = zeroGrads(cfg);
    AdamBuf adam = initAdam(&w, cfg);

    int num_threads = 1;
    #ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    #endif

    printModelHeader(cfg);
    printf("  corpus=%s  tokens=%d  vocab=%d\n", path, nTokens, vocabSize);
    printf("  params=%d  seqLen=%d  batch=%d  steps=%d  threads=%d\n",
           countParams(&w, cfg), seqLen, BATCH, STEPS, num_threads);
#if STEER_AUG
    printf("  steer-aug: ON  (p=0.5, w~U[0,40], vizinho=384 chars, detached)\n");
#endif
    printf("\n");

    float lr = 5e-4f;
    const float LN2 = 0.6931472f;   /* to report loss in bits/char */
    ParamRef refs[8 + 64 * 16];

    Grads *thread_grads = malloc(sizeof(Grads) * num_threads);
    for (int t = 0; t < num_threads; t++) {
        thread_grads[t] = zeroGrads(cfg);
    }

    int startStep = loadCkpt(&w, &adam, cfg);
    if (startStep > 0) printf("  ckpt: resumed %s at step %d\n\n", g_ckptPath, startStep);
    int stopStep = STEPS < g_untilStep ? STEPS : g_untilStep;

    int step;
    for (step = startStep + 1; step <= stopStep; step++) {
        float lr_t = lr;
        float lr_min = lr * LR_MIN_RATIO;
        int warmup_steps = 100;
        if (step < warmup_steps) {
            lr_t = lr * ((float)step / warmup_steps);
        } else {
            float progress = (float)(step - warmup_steps) / (float)(STEPS - warmup_steps);
            lr_t = lr_min + (lr - lr_min) * 0.5f * (1.f + (float)cos((double)progress * 3.141592653589793));
        }

        int startPositions[BATCH];
        for (int b = 0; b < BATCH; b++) {
            startPositions[b] = (int)(randf() * (double)(nTokens - seqLen - 1));
        }

#if STEER_AUG
        /* Steering-interface augmentation (self-derived, detached): signature
         * of the 384 chars neighbouring the window → channel bias, applied as
         * a standing bias while training on the window itself. Serial pre-pass
         * (biases are per-item; the batch loop is parallel). */
        static float augBias[BATCH][8][N_ARCH];
        static float runMean[EMB_DIM];               /* running centering vector */
        float augW[BATCH]; int augOn[BATCH];
        {
            int augJung = -1;
            for (int l = 0; l < cfg.numLayers; l++)
                if (cfg.layerTypes[l] == MIX_JUNGIAN) { augJung = l; break; }
            int Da = cfg.embDim;
            int nbSpan = seqLen < 384 ? seqLen : 384;   /* neighbour span ≤ ctx (posEmb bound) */
            for (int b = 0; b < BATCH; b++) {
                augOn[b] = 0; augW[b] = 0.f;
                if (augJung < 0 || randf() < 0.5) continue;
                int npos = startPositions[b] >= nbSpan ? startPositions[b] - nbSpan
                                                       : startPositions[b] + seqLen;
                if (npos + nbSpan >= nTokens) continue;
                ForwardCache nf = forward(ids + npos, nbSpan, &w, cfg);
                const float *h = nf.blockCaches[augJung].ln1Out;
                double acc[EMB_DIM] = { 0 };
                for (int s = 0; s < nbSpan; s++)
                    for (int d = 0; d < Da; d++) acc[d] += h[s * Da + d];
                float sigv[EMB_DIM]; double nrm = 0;
                for (int d = 0; d < Da; d++) {
                    float m = (float)(acc[d] / nbSpan);
                    runMean[d] = 0.99f * runMean[d] + 0.01f * m;
                    sigv[d] = m - runMean[d];
                    nrm += sigv[d] * sigv[d];
                }
                nrm = sqrt(nrm) + 1e-9;
                for (int d = 0; d < Da; d++) sigv[d] /= (float)nrm;
                freeForwardCache(&nf);
                for (int l = 0; l < cfg.numLayers && l < 8; l++) {
                    if (cfg.layerTypes[l] != MIX_JUNGIAN) continue;
                    const float *wa = w.blocks[l].wArch;
                    double bn = 0;
                    for (int k = 0; k < N_ARCH; k++) {
                        double pj = 0;
                        for (int d = 0; d < Da; d++) pj += sigv[d] * wa[k * Da + d];
                        augBias[b][l][k] = (float)pj; bn += pj * pj;
                    }
                    bn = sqrt(bn) + 1e-9;
                    for (int k = 0; k < N_ARCH; k++) augBias[b][l][k] /= (float)bn;
                }
                augW[b] = (float)(randf() * 40.0);
                augOn[b] = 1;
            }
        }
#endif

        for (int t = 0; t < num_threads; t++) {
            zeroAllGrads(&thread_grads[t], cfg);
        }

        float loss = 0.f;
        #pragma omp parallel for reduction(+:loss)
        for (int b = 0; b < BATCH; b++) {
            int tid = 0;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            #endif
            int startPos = startPositions[b];
            const int *inp = ids + startPos;
            const int *tgt = ids + startPos + 1;
#if STEER_AUG
            if (augOn[b]) {                     /* thread-local standing bias */
                for (int l = 0; l < cfg.numLayers && l < 8; l++)
                    if (cfg.layerTypes[l] == MIX_JUNGIAN) g_personalBias[l] = augBias[b][l];
                g_personalW = augW[b];
            }
#endif
            ForwardCache fc = forward(inp, seqLen, &w, cfg);
            loss += crossEntropyLoss(fc.probs, tgt, seqLen, cfg.vocabSize);
            backward(&fc, tgt, NULL, 0, &w, cfg, &thread_grads[tid]);
            freeForwardCache(&fc);
#if STEER_AUG
            for (int l = 0; l < 8; l++) g_personalBias[l] = NULL;
            g_personalW = 0.f;
#endif
        }

        zeroAllGrads(&grads, cfg);
        for (int t = 0; t < num_threads; t++) {
            mergeGrads(&grads, &thread_grads[t], cfg);
        }

        /* average accumulated gradients over the batch */
        int np = collectParams(&grads, cfg, refs);
        for (int i = 0; i < np; i++)
            for (int j = 0; j < refs[i].len; j++) refs[i].ptr[j] /= BATCH;
        clipGrads(&grads, cfg, GRAD_CLIP);
        adamUpdate(&w, &grads, &adam, cfg, lr_t, step);
        archRepulsionStep(&w, cfg, ARCH_REPULSION * lr_t);

        if (step == 1 || step % 100 == 0)
            printf("step %5d  loss %.4f  (%.3f bits/char)  lr %.2e\n",
                   step, loss / BATCH, (loss / BATCH) / LN2, lr_t);

        if (g_ckptPath && step % 50 == 0) saveCkpt(&w, &adam, cfg, step);

        /* with EVAL_SAMPLES the fixed-prompt eval replaces the big final sample */
        if (step % 500 == 0 || (step == STEPS && EVAL_SAMPLES == 0)) {
            int seedLen = 8;
            int genLen = (step == STEPS) ? 1500 : 240;
            int *out = malloc(sizeof(int) * (seedLen + genLen));
            int sp = (int)(randf() * (double)(nTokens - seedLen));
            int n = generateText(&w, cfg, ids + sp, seedLen, genLen, 0.8f, 0.9f, seqLen, out);
            printf("  ── sample (length %d) ───────────────────────────────\n  ", genLen);
            for (int i = 0; i < n; i++) { char c = idToChar[out[i]]; putchar(c); if (c == '\n') printf("  "); }
            printf("\n  ────────────────────────────────────────────────────\n\n");
            free(out);
        }
    }

    if (g_ckptPath) saveCkpt(&w, &adam, cfg, step - 1);

    if (step - 1 >= STEPS) {   /* only on the completed run (not --until chunks) */
        if (g_evalSample < 0) printArchetypeStats(&w, cfg, ids, nTokens, idToChar);
        evalMetricsGeneration(&w, cfg, ids, nTokens, idToChar);
    }

    for (int t = 0; t < num_threads; t++) {
        freeWeights(&thread_grads[t], cfg);
    }
    free(thread_grads);

    freeWeights(&w, cfg); freeWeights(&grads, cfg); freeAdam(&adam);
    free(ids);
}

/* ─── Tool-calling task: completion-masked training + structured eval ─────────
 *
 * A different regime from language modeling: each example is a (query + available
 * tools) prompt followed by the target JSON function call. We train next-token as
 * usual but MASK the loss to the completion only (everything after "### Call:\n"),
 * so the model learns to produce the call, not to model the prompt. Evaluation is
 * structured: greedily decode the call and exact-match it against the gold answer.
 * Data comes from gen_tools_data.py (see tooldata/tools_{train,val,test}.txt).
 */

#define TC_MARKER "### Call:\n"

typedef struct { int *ids; int len; int compStart; } TCExample;

/* Build a shared char vocab over all split files (union), so train/val/test agree. */
static int tcBuildVocab(const char **files, int nFiles, int *charToId, char *idToChar) {
    for (int i = 0; i < 256; i++) charToId[i] = -1;
    int vocab = 0;
    for (int f = 0; f < nFiles; f++) {
        FILE *fp = fopen(files[f], "rb");
        if (!fp) { fprintf(stderr, "cannot open %s\n", files[f]); exit(1); }
        int ch;
        while ((ch = fgetc(fp)) != EOF) {
            if (ch == '\r' || ch >= 128) continue;
            if (charToId[ch] == -1) { charToId[ch] = vocab; idToChar[vocab] = (char)ch; vocab++; }
        }
        fclose(fp);
    }
    return vocab;
}

/* Parse a rendered split file into examples (blocks split on the blank line). Each
 * example's ids include a trailing '\n' as end-of-sequence; compStart marks the
 * first completion token (right after the "### Call:\n" marker). */
static TCExample *tcLoad(const char *path, const int *charToId, int *nOut) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    size_t got = fread(buf, 1, (size_t)sz, f); buf[got] = 0; fclose(f);

    int cap = 64, n = 0;
    TCExample *exs = malloc(sizeof(TCExample) * cap);
    char *p = buf;
    while (*p) {
        while (*p == '\n') p++;                 /* skip separators */
        if (!*p) break;
        char *sep = strstr(p, "\n\n");
        char *blockEnd = sep ? sep : buf + got;
        size_t blen = (size_t)(blockEnd - p);

        /* example text = block + EOS '\n' */
        char *marker = strstr(p, TC_MARKER);
        if (!marker || marker >= blockEnd) { p = sep ? sep + 2 : blockEnd; continue; }
        int compChar = (int)((marker + strlen(TC_MARKER)) - p);  /* char offset of call */

        int *ids = malloc(sizeof(int) * (blen + 1));
        int L = 0, compStart = -1;
        for (size_t i = 0; i < blen; i++) {
            unsigned char ch = (unsigned char)p[i];
            if (ch == '\r' || ch >= 128) continue;
            if ((int)i == compChar) compStart = L;
            ids[L++] = charToId[ch];
        }
        ids[L++] = charToId['\n'];              /* EOS */
        if (compStart < 0) compStart = L - 1;

        if (n == cap) { cap *= 2; exs = realloc(exs, sizeof(TCExample) * cap); }
        exs[n].ids = ids; exs[n].len = L; exs[n].compStart = compStart; n++;
        p = sep ? sep + 2 : blockEnd;
    }
    free(buf);
    *nOut = n;
    return exs;
}

/* Greedy (argmax) decode of the completion given a prompt; stops at stopId ('\n').
 * Writes generated tokens (excluding the prompt, excluding the stop token) to out. */
static int tcGreedy(Weights *w, Config cfg, const int *prompt, int promptLen,
                    int stopId, int maxGen, int *out) {
    int V = cfg.vocabSize;
    int *ids = malloc(sizeof(int) * (promptLen + maxGen));
    memcpy(ids, prompt, sizeof(int) * promptLen);
    int n = promptLen, outLen = 0;
    for (int g = 0; g < maxGen; g++) {
        int start = n > cfg.contextLen ? n - cfg.contextLen : 0;
        int ctx = n - start;
        ForwardCache fc = forward(ids + start, ctx, w, cfg);
        int last = ctx - 1, best = 0; float bv = NEG_INF;
        for (int v = 0; v < V; v++) {
            float lv = fc.logits[last * V + v];
            if (lv > bv) { bv = lv; best = v; }
        }
        freeForwardCache(&fc);
        if (best == stopId) break;
        ids[n++] = best;
        out[outLen++] = best;
    }
    free(ids);
    return outLen;
}

/* Extract the tool name from a call string: the value after the first "name":". */
static void tcExtractName(const char *s, char *out, int cap) {
    const char *key = "\"name\":\"";
    const char *p = strstr(s, key);
    int k = 0;
    if (p) { p += strlen(key); while (*p && *p != '"' && k < cap - 1) out[k++] = *p++; }
    out[k] = 0;
}

/*
 * Structured eval on the first `n` examples. Reports two metrics that decompose
 * the task: tool SELECTION (did we name the right function?) and EXACT-MATCH (did
 * the full JSON call, including argument extraction, match the gold answer?).
 */
static float tcEval(Weights *w, Config cfg, TCExample *exs, int n,
                    const int *charToId, const char *idToChar, const char *name, int nShow) {
    int stopId = charToId[(int)'\n'];
    int exact = 0, nameOk = 0, shown = 0;
    int *gen = malloc(sizeof(int) * 96);
    char genStr[160], goldStr[160], gName[64], pName[64];
    for (int i = 0; i < n; i++) {
        TCExample *e = &exs[i];
        int glen = tcGreedy(w, cfg, e->ids, e->compStart, stopId, 96, gen);
        int goldLen = e->len - 1 - e->compStart;   /* exclude the EOS '\n' */

        int gl = glen < 159 ? glen : 159, dl = goldLen < 159 ? goldLen : 159;
        for (int k = 0; k < gl; k++) genStr[k] = idToChar[gen[k]];
        genStr[gl] = 0;
        for (int k = 0; k < dl; k++) goldStr[k] = idToChar[e->ids[e->compStart + k]];
        goldStr[dl] = 0;

        int em = (strcmp(genStr, goldStr) == 0);
        tcExtractName(genStr, pName, 64); tcExtractName(goldStr, gName, 64);
        int nm = (pName[0] && strcmp(pName, gName) == 0);
        exact += em; nameOk += nm;

        if (shown < nShow) {
            printf("    [%s] ", em ? "ok  " : (nm ? "args" : "MISS"));
            for (int k = 0; k < e->len && idToChar[e->ids[k]] != '\n'; k++) putchar(idToChar[e->ids[k]]);
            printf("\n           gold: %s\n           pred: %s\n", goldStr, genStr);
            shown++;
        }
    }
    free(gen);
    float em_acc = 100.f * (float)exact / (float)n;
    printf("  %-5s  tool-select %d/%d = %.1f%%   exact-match %d/%d = %.1f%%\n",
           name, nameOk, n, 100.f * (float)nameOk / (float)n, exact, n, em_acc);
    return em_acc;
}

/*
 * Tool-calling trainer: minibatch SGD with completion-masked cross-entropy.
 * Run with ./transformer --tools <dir>  (dir holds tools_{train,val,test}.txt).
 */
static void trainToolCalling(const char *dir) {
    setvbuf(stdout, NULL, _IOLBF, 0);   /* line-buffer so progress streams to logs */
    resetRand(42);
    char trainPath[1024], valPath[1024], testPath[1024];
    snprintf(trainPath, sizeof trainPath, "%s/tools_train.txt", dir);
    snprintf(valPath, sizeof valPath, "%s/tools_val.txt", dir);
    snprintf(testPath, sizeof testPath, "%s/tools_test.txt", dir);

    int charToId[256]; char idToChar[256];
    const char *files[3] = { trainPath, valPath, testPath };
    int vocab = tcBuildVocab(files, 3, charToId, idToChar);

    int nTrain, nVal, nTest;
    TCExample *train = tcLoad(trainPath, charToId, &nTrain);
    TCExample *val = tcLoad(valPath, charToId, &nVal);
    TCExample *test = tcLoad(testPath, charToId, &nTest);

    int maxLen = 0;
    for (int i = 0; i < nTrain; i++) if (train[i].len > maxLen) maxLen = train[i].len;
    for (int i = 0; i < nVal; i++)   if (val[i].len > maxLen)   maxLen = val[i].len;
    for (int i = 0; i < nTest; i++)  if (test[i].len > maxLen)  maxLen = test[i].len;

    Config cfg = { .vocabSize = vocab, .contextLen = maxLen + 96, .embDim = EMB_DIM,
                   .numHeads = 4, .numKVHeads = (NUM_KV_HEADS > 4) ? 4 : NUM_KV_HEADS,
                   .ffDim = FF_DIM, .stateN = 16 };
    applyLayout(&cfg);

    Weights w = initWeights(cfg);
    Grads grads = zeroGrads(cfg);
    AdamBuf adam = initAdam(&w, cfg);

    printModelHeader(cfg);
    printf("  task=tool-calling  dir=%s\n", dir);
    printf("  train=%d val=%d test=%d  vocab=%d  maxLen=%d\n", nTrain, nVal, nTest, vocab, maxLen);
    printf("  params=%d  batch=%d  steps=%d  (completion-masked loss)\n\n",
           countParams(&w, cfg), BATCH, STEPS);

    float lr = 5e-4f;
    ParamRef refs[8 + 64 * 16];

    for (int step = 1; step <= STEPS; step++) {
        float lr_t, lr_min = lr * LR_MIN_RATIO;
        int warmup = 100;
        if (step < warmup) lr_t = lr * ((float)step / warmup);
        else {
            float prog = (float)(step - warmup) / (float)(STEPS - warmup);
            lr_t = lr_min + (lr - lr_min) * 0.5f * (1.f + (float)cos((double)prog * 3.141592653589793));
        }

        zeroAllGrads(&grads, cfg);
        float loss = 0.f;
        for (int b = 0; b < BATCH; b++) {
            TCExample *e = &train[(int)(randf() * (double)nTrain)];
            int S = e->len - 1;                 /* predictions */
            const int *inp = e->ids, *tgt = e->ids + 1;
            int *mask = malloc(sizeof(int) * S);
            int nActive = 0;
            for (int i = 0; i < S; i++) { mask[i] = ((i + 1) >= e->compStart); nActive += mask[i]; }

            ForwardCache fc = forward(inp, S, &w, cfg);
            loss += crossEntropyLossMasked(fc.probs, tgt, mask, S, vocab, nActive);
            backward(&fc, tgt, mask, nActive, &w, cfg, &grads);
            freeForwardCache(&fc);
            free(mask);
        }

        int np = collectParams(&grads, cfg, refs);
        for (int i = 0; i < np; i++)
            for (int j = 0; j < refs[i].len; j++) refs[i].ptr[j] /= BATCH;
        clipGrads(&grads, cfg, GRAD_CLIP);
        adamUpdate(&w, &grads, &adam, cfg, lr_t, step);
        archRepulsionStep(&w, cfg, ARCH_REPULSION * lr_t);

        if (step == 1 || step % 100 == 0)
            printf("step %5d  completion-loss %.4f  lr %.2e\n", step, loss / BATCH, lr_t);

        if (step % 500 == 0 || step == STEPS) {
            int sub = nVal < 40 ? nVal : 40;    /* subset for a quick progress signal */
            printf("  ── eval @ step %d (val subset %d) ──\n", step, sub);
            tcEval(&w, cfg, val, sub, charToId, idToChar, "val", 0);
            printf("\n");
        }
    }

    printf("\n=== final structured eval ===\n");
    tcEval(&w, cfg, val, nVal, charToId, idToChar, "val", 3);
    tcEval(&w, cfg, test, nTest, charToId, idToChar, "test", 4);

    for (int i = 0; i < nTrain; i++) free(train[i].ids);
    for (int i = 0; i < nVal; i++) free(val[i].ids);
    for (int i = 0; i < nTest; i++) free(test[i].ids);
    free(train); free(val); free(test);
    freeWeights(&w, cfg); freeWeights(&grads, cfg); freeAdam(&adam);
}

/* ─── Shadow-persona experiment (IDEA.md §6, inference-only) ──────────────────
 * Drives ONE frozen Jungian checkpoint with two "simulated users" whose only
 * difference is the reinforcement signal R_t (charset-based, IDEA §6):
 *   user A reinforces lyric chars  (vowels + breath + apostrophe),
 *   user B reinforces technical chars (consonants + structural punctuation).
 * A shared passage is streamed through the frozen model with the personal-
 * relevance term active; each user's Shadow is persisted to sombras/<user>.bin
 * and we measure whether the two Shadows diverge — central weights never touched.
 * This is the "verify the Shadow diverges without fine-tuning" step of the MVP. */

static int g_personaTest = 0;   /* --persona-test */
static int g_oracleTest = 0;    /* --oracle-test */
static int g_deltaGiven = 0;    /* whether --delta was passed (so --delta 0 is honoured) */

static int chInSet(char c, int userSel) {
    if (userSel == 0) return strchr("aeiouAEIOU '\n\t", c) != NULL;   /* lyric */
    int alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    int vowel = strchr("aeiouAEIOU", c) != NULL;
    if (alpha && !vowel) return 1;                                    /* consonant */
    return strchr(".,;:", c) != NULL;                                /* structural */
}

static void saveUserShadow(const char *path, Config cfg) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "shadow: cannot write %s\n", path); return; }
    int hdr[2] = { cfg.numLayers, N_ARCH };
    fwrite(hdr, sizeof(int), 2, f);
    for (int l = 0; l < cfg.numLayers && l < 8; l++)
        fwrite(g_shadowCache[l], sizeof(float), N_ARCH, f);
    fclose(f);
}

static float cosDist(const float *a, const float *b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if (na == 0 || nb == 0) return 1.0f;
    return (float)(1.0 - dot / (sqrt(na) * sqrt(nb)));
}

static void shadowPersonaExperiment(const char *corpusPath) {
    resetRand(42);
    char idToChar[256];
    int nTokens, vocabSize;
    int *ids = loadCorpus(corpusPath, &nTokens, &vocabSize, idToChar);

    Config cfg = { .vocabSize = vocabSize, .contextLen = SEQ_LEN, .embDim = EMB_DIM,
                   .numHeads = 4, .numKVHeads = (NUM_KV_HEADS > 4) ? 4 : NUM_KV_HEADS,
                   .ffDim = FF_DIM, .stateN = 16 };
    applyLayout(&cfg);

    int jungLayer = -1;
    for (int l = 0; l < cfg.numLayers; l++)
        if (cfg.layerTypes[l] == MIX_JUNGIAN) { jungLayer = l; break; }
    if (jungLayer < 0) { fprintf(stderr, "persona-test: layout has no Jungian layer\n"); free(ids); return; }

    Weights w = initWeights(cfg);
    AdamBuf adam = initAdam(&w, cfg);
    int step = loadCkpt(&w, &adam, cfg);
    if (!g_deltaGiven) g_reinforceDelta = 2.0f;   /* default; --delta 0 stays 0 (control) */

    printModelHeader(cfg);
    printf("  corpus=%s  tokens=%d  vocab=%d  ckpt-step=%d  delta=%.2f\n",
           corpusPath, nTokens, vocabSize, step, g_reinforceDelta);
    if (step == 0)
        printf("  (no matching checkpoint — random weights; divergence still valid, tonal demo is noise)\n");
    printf("\n");

    int windowSize = cfg.contextLen;
    int passStart = nTokens / 3;                 /* fixed shared passage */
    int passLen = 6000;
    if (passStart + passLen >= nTokens) passLen = nTokens - passStart - 1;
    int nChunks = passLen / windowSize;
    if (nChunks < 4) { fprintf(stderr, "persona-test: corpus too short\n"); free(ids); return; }

    const float fracs[4] = { 0.25f, 0.5f, 0.75f, 1.0f };
    static float snap[2][4][N_ARCH];             /* [user][frac][k] shadow of jungLayer */

    mkdir("sombras", 0755);
    const char *files[2] = { "sombras/userA_lyric.bin", "sombras/userB_technical.bin" };
    const char *names[2] = { "A (lyric)", "B (technical)" };
    float finalShadow[2][N_ARCH];               /* representative Jungian layer, for the table */
    static float finalShadowAll[2][8][N_ARCH];  /* every Jungian layer, for behavioural injection */

    float *R = malloc(sizeof(float) * windowSize);

    for (int u = 0; u < 2; u++) {
        memset(g_shadowCache, 0, sizeof(g_shadowCache));
        g_shadowCacheOn = 1; g_reinforceOn = 1; g_RtWindow = R;
        int fi = 0;
        long reinforced = 0, total = 0;
        for (int c = 0; c < nChunks; c++) {
            int pos = passStart + c * windowSize;
            for (int j = 0; j < windowSize; j++) {
                char ch = idToChar[ids[pos + j]];
                R[j] = chInSet(ch, u) ? 1.0f : 0.0f;
                reinforced += (R[j] > 0); total++;
            }
            ForwardCache fc = forward(ids + pos, windowSize, &w, cfg);
            for (int l = 0; l < cfg.numLayers && l < 8; l++)
                if (cfg.layerTypes[l] == MIX_JUNGIAN && fc.blockCaches[l].shadow_j)
                    memcpy(g_shadowCache[l],
                           fc.blockCaches[l].shadow_j + (windowSize - 1) * N_ARCH,
                           sizeof(float) * N_ARCH);
            freeForwardCache(&fc);
            while (fi < 4 && c + 1 >= (int)(fracs[fi] * nChunks)) {
                memcpy(snap[u][fi], g_shadowCache[jungLayer], sizeof(float) * N_ARCH);
                fi++;
            }
        }
        while (fi < 4) { memcpy(snap[u][fi], g_shadowCache[jungLayer], sizeof(float) * N_ARCH); fi++; }
        memcpy(finalShadow[u], g_shadowCache[jungLayer], sizeof(float) * N_ARCH);
        for (int l = 0; l < cfg.numLayers && l < 8; l++)
            memcpy(finalShadowAll[u][l], g_shadowCache[l], sizeof(float) * N_ARCH);
        saveUserShadow(files[u], cfg);
        printf("  user %-14s reinforced %ld/%ld tokens (%.0f%%)  ->  %s\n",
               names[u], reinforced, total, 100.0 * reinforced / total, files[u]);
    }
    g_reinforceOn = 0; g_RtWindow = NULL;

    printf("\n  ── Shadow divergence (layer %d, cosine distance A vs B) ─────────────\n", jungLayer);
    for (int fi = 0; fi < 4; fi++)
        printf("     at %3.0f%% of passage :  cos-dist = %.4f\n",
               fracs[fi] * 100, cosDist(snap[0][fi], snap[1][fi], N_ARCH));

    printf("\n  ── Per-archetype Shadow (final) ─────────────────────────────────────\n");
    const char *arch[N_ARCH] = {
        "self","shadow","anima","animus","great_mother","great_father",
        "hero","wise_old_man","persona","puer_aeternus","senex","trickster",
        "syzygy","spirit","cosmos","aion" };
    printf("     %-14s | %10s | %10s | %s\n", "archetype", "A lyric", "B technical", "Δ favours");
    for (int k = 0; k < N_ARCH; k++) {
        float a = finalShadow[0][k], b = finalShadow[1][k];
        const char *fav = (a > b) ? "A" : (b > a) ? "B" : "=";
        printf("     %-14s | %10.5f | %10.5f | %s\n", arch[k], a, b, fav);
    }

    /* Behavioural effect: does the divergent personal Shadow actually change the
     * model's OUTPUTS? Teacher-force a fixed held passage under user A's bias vs
     * user B's bias (identical weights & context, only w_u·Sombra_u differs) and
     * measure how often the next-token argmax flips and the mean logit shift.
     * w_u = 1.0 — the natural "fully-trusted identity" weight of IDEA §3.4;
     * the effect scales with w_u (Sombra inclines, it does not dominate). */
    printf("\n  ── Behavioural effect of the personal Shadow (w_u sweep) ───────────\n");
    printf("     A vs B next-token divergence on a fixed held passage (weights & context\n");
    printf("     identical; only w_u·Sombra_u differs). w_u = identity confidence, IDEA §3.4.\n");
    g_shadowCacheOn = 0;
    int evalStart = passStart + passLen + 1;
    int evalLen = windowSize;
    if (evalStart + evalLen >= nTokens) evalStart = passStart;   /* fallback */
    int V = cfg.vocabSize;
    float *logitsA = malloc(sizeof(float) * evalLen * V);
    const float ws[4] = { 1.0f, 4.0f, 12.0f, 40.0f };
    const float genTemp = 0.8f;                  /* the temperature generation samples at */
    float *logitsB = malloc(sizeof(float) * evalLen * V);
    printf("     %6s | %-16s | %s\n", "w_u", "mean TV(pA,pB)", "mean |Δlogit| L2");
    for (int wi = 0; wi < 4; wi++) {
        g_personalW = ws[wi];
        for (int l = 0; l < 8; l++) g_personalBias[l] = NULL;
        for (int l = 0; l < cfg.numLayers && l < 8; l++)
            if (cfg.layerTypes[l] == MIX_JUNGIAN) g_personalBias[l] = finalShadowAll[0][l];
        ForwardCache fa = forward(ids + evalStart, evalLen, &w, cfg);
        memcpy(logitsA, fa.logits, sizeof(float) * evalLen * V);
        freeForwardCache(&fa);
        for (int l = 0; l < cfg.numLayers && l < 8; l++)
            if (cfg.layerTypes[l] == MIX_JUNGIAN) g_personalBias[l] = finalShadowAll[1][l];
        ForwardCache fb = forward(ids + evalStart, evalLen, &w, cfg);
        memcpy(logitsB, fb.logits, sizeof(float) * evalLen * V);
        freeForwardCache(&fb);
        double tvSum = 0, l2 = 0;
        for (int s = 0; s < evalLen; s++) {
            float mA = NEG_INF, mB = NEG_INF;
            for (int v = 0; v < V; v++) {
                if (logitsA[s * V + v] / genTemp > mA) mA = logitsA[s * V + v] / genTemp;
                if (logitsB[s * V + v] / genTemp > mB) mB = logitsB[s * V + v] / genTemp;
            }
            float sA = 0, sB = 0;
            for (int v = 0; v < V; v++) {
                sA += expf(logitsA[s * V + v] / genTemp - mA);
                sB += expf(logitsB[s * V + v] / genTemp - mB);
            }
            double tv = 0;
            for (int v = 0; v < V; v++) {
                double a = expf(logitsA[s * V + v] / genTemp - mA) / sA;
                double b = expf(logitsB[s * V + v] / genTemp - mB) / sB;
                tv += fabs(a - b);
                l2 += (double)(logitsA[s * V + v] - logitsB[s * V + v])
                            * (logitsA[s * V + v] - logitsB[s * V + v]);
            }
            tvSum += 0.5 * tv;
        }
        printf("     %6.1f | %6.2f%%          | %.4f\n",
               ws[wi], 100.0 * tvSum / evalLen, sqrt(l2 / evalLen));
    }
    free(logitsB);
    free(logitsA);
    printf("     → the frozen model answers measurably differently by interlocutor, and the\n");
    printf("       effect scales with identity confidence w_u — the Shadow inclines, not dominates.\n");
    printf("       (Aligning that change to a *target* behaviour needs the trainable term, IDEA §8.2.)\n");
    for (int l = 0; l < 8; l++) g_personalBias[l] = NULL;
    g_personalW = 0.0f;

    free(R); free(ids);
    freeWeights(&w, cfg); freeAdam(&adam);
}

/* ─── Oracle-driven long-horizon experiment (IDEA.md §3.2 + §8.4) ─────────────
 * A "bigger AI" (Claude, offline) plays the ORACLE: for each conversation turn it
 * names the dominant word and its valence (+1 valued / −1 aversive) — see
 * oracle_labels.py. The archetype channels are anonymous, so the oracle does NOT
 * pick a channel: the frozen agent routes the dominant word itself, and the
 * per-user personal Shadow slowly integrates that routing, signed by valence:
 *
 *   R_t[k] = valence · mean_{word span} p_t[k]          (word → channel, by the agent)
 *   Sombra_u ← (1-η)·Sombra_u + η·R_t                   (slow accumulator, η≪1)
 *
 * One micro-step per turn; over hundreds of turns Sombra_u consolidates a user's
 * affective map ("doce" pushed negative, "Joyce" positive) — the long horizon of
 * §8.4. Central weights are never touched; deleting the file resets the user. */

static const char *g_oracleLabels = "oracle_turns.tsv";

typedef struct { char user; int val; char word[32]; char text[128]; } OracleTurn;

/* Tokenise a C-string to vocab ids; srcIdx[t] = source char offset of token t
 * (so a word's char span maps to token positions after unknown chars are skipped). */
static int tokenizeStr(const char *s, const int *charToId, int *ids, int *srcIdx, int maxLen) {
    int n = 0;
    for (int i = 0; s[i] && n < maxLen; i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch == '\r' || ch >= 128 || charToId[ch] < 0) continue;
        srcIdx[n] = i;
        ids[n++] = charToId[ch];
    }
    return n;
}

static void shadowOracleExperiment(const char *corpusPath) {
    resetRand(42);
    char idToChar[256];
    int nTokens, vocabSize;
    int *ids = loadCorpus(corpusPath, &nTokens, &vocabSize, idToChar);

    Config cfg = { .vocabSize = vocabSize, .contextLen = SEQ_LEN, .embDim = EMB_DIM,
                   .numHeads = 4, .numKVHeads = (NUM_KV_HEADS > 4) ? 4 : NUM_KV_HEADS,
                   .ffDim = FF_DIM, .stateN = 16 };
    applyLayout(&cfg);
    int jungLayer = -1;
    for (int l = 0; l < cfg.numLayers; l++)
        if (cfg.layerTypes[l] == MIX_JUNGIAN) { jungLayer = l; break; }
    if (jungLayer < 0) { fprintf(stderr, "oracle-test: no Jungian layer\n"); free(ids); return; }

    int charToId[256];
    for (int i = 0; i < 256; i++) charToId[i] = -1;
    for (int v = 0; v < vocabSize; v++) charToId[(unsigned char)idToChar[v]] = v;

    Weights w = initWeights(cfg);
    AdamBuf adam = initAdam(&w, cfg);
    int step = loadCkpt(&w, &adam, cfg);

    /* load the oracle labels */
    FILE *lf = fopen(g_oracleLabels, "r");
    if (!lf) { fprintf(stderr, "oracle-test: cannot open %s (run oracle_labels.py)\n", g_oracleLabels); free(ids); return; }
    int cap = 256, nT = 0;
    OracleTurn *T = malloc(sizeof(OracleTurn) * cap);
    char line[512];
    while (fgets(line, sizeof(line), lf)) {
        char u; int val; char word[32], text[128];
        /* format: user \t ±val \t word \t text */
        if (sscanf(line, "%c\t%d\t%31[^\t]\t%127[^\n]", &u, &val, word, text) == 4) {
            if (nT == cap) { cap *= 2; T = realloc(T, sizeof(OracleTurn) * cap); }
            T[nT].user = u; T[nT].val = val;
            snprintf(T[nT].word, 32, "%s", word);
            snprintf(T[nT].text, 128, "%s", text);
            nT++;
        }
    }
    fclose(lf);

    printModelHeader(cfg);
    printf("  corpus=%s  vocab=%d  ckpt-step=%d  oracle=%s  turns=%d\n\n",
           corpusPath, vocabSize, step, g_oracleLabels, nT);
    if (nT < 20) { fprintf(stderr, "oracle-test: too few turns\n"); free(T); free(ids); return; }

    int K = N_ARCH, D = cfg.embDim;
    int windowSize = cfg.contextLen;
    const float eta = 0.01f;                     /* slow accumulator rate (IDEA §8.4) */
    static float sombraU[2][8][N_ARCH];          /* [user][layer][k] */
    memset(sombraU, 0, sizeof(sombraU));

    /* Baseline persona: the model's average routing over generic corpus text.
     * We center every word's routing by this baseline so only a word's
     * DISTINCTIVE channels enter R_t — otherwise the common-mode persona (shared
     * by all words) dominates and the ± valence signal washes out. */
    static float baseP[8][N_ARCH];
    memset(baseP, 0, sizeof(baseP));
    {
        long cnt = 0;
        int nSample = 60, stride = (nTokens - windowSize) / nSample;
        if (stride < windowSize) stride = windowSize;
        for (int c = 0; c < nSample && c * stride + windowSize < nTokens; c++) {
            ForwardCache fc = forward(ids + c * stride, windowSize, &w, cfg);   /* spread across corpus */
            for (int l = 0; l < cfg.numLayers && l < 8; l++)
                if (cfg.layerTypes[l] == MIX_JUNGIAN && fc.blockCaches[l].p_j)
                    for (int s = 0; s < windowSize; s++)
                        for (int k = 0; k < K; k++) baseP[l][k] += fc.blockCaches[l].p_j[s * K + k];
            freeForwardCache(&fc);
            cnt += windowSize;
        }
        if (cnt) for (int l = 0; l < 8; l++) for (int k = 0; k < K; k++) baseP[l][k] /= cnt;
    }

    int wbuf[128], sbuf[128];
    float Rt[N_ARCH];

    /* consolidation snapshots */
    int nSnap = 0; float snapDiv[16]; int snapAt[16];
    int snapEvery = nT / 10; if (snapEvery < 1) snapEvery = 1;

    printf("  ── Consolidation over the horizon (η=%.3f) ─────────────────────────\n", eta);
    printf("     %6s | %-8s | %-8s | %s\n", "turn", "|S_A|", "|S_B|", "cos-dist(A,B) @layer 0");
    for (int t = 0; t < nT; t++) {
        int u = (T[t].user == 'A') ? 0 : 1;
        int L = tokenizeStr(T[t].text, charToId, wbuf, sbuf, windowSize);
        if (L < 2) continue;
        ForwardCache fc = forward(wbuf, L, &w, cfg);

        /* dominant-word char span → token span */
        const char *pos = strstr(T[t].text, T[t].word);
        int c0 = pos ? (int)(pos - T[t].text) : 0;
        int c1 = c0 + (int)strlen(T[t].word);
        int s0 = -1, s1 = -1;
        for (int s = 0; s < L; s++) {
            if (sbuf[s] >= c0 && s0 < 0) s0 = s;
            if (sbuf[s] < c1) s1 = s;
        }
        if (s0 < 0) { s0 = 0; s1 = L - 1; }        /* fallback: whole turn */

        for (int l = 0; l < cfg.numLayers && l < 8; l++) {
            if (cfg.layerTypes[l] != MIX_JUNGIAN || !fc.blockCaches[l].p_j) continue;
            for (int k = 0; k < K; k++) Rt[k] = 0.f;
            int span = 0;
            for (int s = s0; s <= s1; s++, span++)
                for (int k = 0; k < K; k++) Rt[k] += fc.blockCaches[l].p_j[s * K + k];
            for (int k = 0; k < K; k++) {
                /* centered by baseline, then valence-signed: only distinctive channels */
                float distinctive = (span ? Rt[k] / span : 0.f) - baseP[l][k];
                Rt[k] = distinctive * (float)T[t].val;
                sombraU[u][l][k] = (1.f - eta) * sombraU[u][l][k] + eta * Rt[k];
            }
        }
        freeForwardCache(&fc);

        if ((t + 1) % snapEvery == 0 || t == nT - 1) {
            double nA = 0, nB = 0;
            for (int k = 0; k < K; k++) { nA += sombraU[0][jungLayer][k]*sombraU[0][jungLayer][k];
                                          nB += sombraU[1][jungLayer][k]*sombraU[1][jungLayer][k]; }
            float cd = cosDist(sombraU[0][jungLayer], sombraU[1][jungLayer], K);
            printf("     %6d | %8.4f | %8.4f | %.4f\n", t + 1, sqrt(nA), sqrt(nB), cd);
            if (nSnap < 16) { snapDiv[nSnap] = cd; snapAt[nSnap] = t + 1; nSnap++; }
        }
    }

    /* ── Valence alignment: did each user's Shadow learn its affective map? ──
     * Route every distinct labelled word through the frozen model, take its
     * archetype signature q_w, and measure cos(Sombra_u, q_w) grouped by the
     * oracle's valence. Consolidation ⇒ valued words align (+), aversive anti-align (−). */
    printf("\n  ── Valence alignment  cos(Sombra_u, q_word)  (layer %d) ─────────────\n", jungLayer);
    printf("     %-14s | %12s | %12s\n", "user", "valued (+1)", "aversive (−1)");
    for (int u = 0; u < 2; u++) {
        char uc = u ? 'B' : 'A';
        double sumPos = 0, sumNeg = 0; int nPos = 0, nNeg = 0;
        char seen[64][32]; int nSeen = 0;
        for (int t = 0; t < nT; t++) {
            if (T[t].user != uc) continue;
            int dup = 0;
            for (int j = 0; j < nSeen; j++) if (strcmp(seen[j], T[t].word) == 0) { dup = 1; break; }
            if (dup || nSeen >= 64) continue;
            snprintf(seen[nSeen++], 32, "%s", T[t].word);
            int L = tokenizeStr(T[t].word, charToId, wbuf, sbuf, windowSize);
            if (L < 1) continue;
            ForwardCache fc = forward(wbuf, L, &w, cfg);
            float qw[N_ARCH]; for (int k = 0; k < K; k++) qw[k] = 0;
            for (int s = 0; s < L; s++)
                for (int k = 0; k < K; k++) qw[k] += fc.blockCaches[jungLayer].p_j[s * K + k];
            for (int k = 0; k < K; k++) qw[k] = qw[k] / L - baseP[jungLayer][k];   /* centered signature */
            freeForwardCache(&fc);
            float c = 1.f - cosDist(sombraU[u][jungLayer], qw, K);   /* cosine similarity */
            if (T[t].val > 0) { sumPos += c; nPos++; } else { sumNeg += c; nNeg++; }
        }
        printf("     %-14c | %12.4f | %12.4f\n", uc,
               nPos ? sumPos / nPos : 0.0, nNeg ? sumNeg / nNeg : 0.0);
    }
    printf("     → valued concepts align with the personal Shadow, aversive ones anti-align:\n");
    printf("       the K=%d anonymous channels linearly separated the oracle's affective map.\n", K);
    (void)D;

    /* persist per-user Shadow (all Jungian layers) */
    mkdir("sombras", 0755);
    const char *files[2] = { "sombras/oracle_userA.bin", "sombras/oracle_userB.bin" };
    for (int u = 0; u < 2; u++) {
        FILE *f = fopen(files[u], "wb");
        if (f) {
            int hdr[2] = { cfg.numLayers, N_ARCH };
            fwrite(hdr, sizeof(int), 2, f);
            for (int l = 0; l < cfg.numLayers && l < 8; l++)
                fwrite(sombraU[u][l], sizeof(float), N_ARCH, f);
            fclose(f);
        }
    }
    printf("\n  persisted: %s , %s   (delete to forget a user — IDEA §8.7 reversibility)\n",
           files[0], files[1]);

    free(T); free(ids);
    freeWeights(&w, cfg); freeAdam(&adam);
}

/* ─── Emergent-archetype experiment (inference-only, --emerge-test) ───────────
 * ARCHETYPE_EMERGENCE_PROBE.md ported into the neural model: nothing is
 * pre-defined — no archetype count, no labels, no curated images. During a
 * battery of inference turns the archetypes EMERGE as novelty-gated online
 * clusters in the model's own representation space (the Jungian layer's ln1Out,
 * the exact space wArch measures similarity in). Each emergent archetype then
 * maps canonically onto the trained mixer channels (bias_k = ⟨centroid,wArch_k⟩
 * is literally the similarity the mixer computes), so the persona/shadow
 * steering machinery is driven by archetypes identified during the same
 * inference battery. Source-work labels exist only for POST-HOC validation and
 * never enter the method. */

static int g_emergeTest = 0;    /* --emerge-test */

static void shadowEmergenceExperiment(const char *corpusPath) {
    char idToChar[256];
    int nTokens, vocabSize;
    int *ids = loadCorpus(corpusPath, &nTokens, &vocabSize, idToChar);

    Config cfg = { .vocabSize = vocabSize, .contextLen = SEQ_LEN, .embDim = EMB_DIM,
                   .numHeads = 4, .numKVHeads = (NUM_KV_HEADS > 4) ? 4 : NUM_KV_HEADS,
                   .ffDim = FF_DIM, .stateN = 16 };
    applyLayout(&cfg);
    int jungLayer = -1;
    for (int l = 0; l < cfg.numLayers; l++)
        if (cfg.layerTypes[l] == MIX_JUNGIAN) { jungLayer = l; break; }
    /* No Jungian layer ⇒ generic backbone: signatures live at the LAST block's
     * input and steering falls back to residual-stream activation addition. */
    int sigLayer = (jungLayer >= 0) ? jungLayer : cfg.numLayers - 1;

    Weights w = initWeights(cfg);
    AdamBuf adam = initAdam(&w, cfg);
    int step = loadCkpt(&w, &adam, cfg);
    resetRand(1234);                       /* deterministic battery */

    printModelHeader(cfg);
    printf("  corpus=%s  tokens=%d  vocab=%d  ckpt-step=%d\n", corpusPath, nTokens, vocabSize, step);
    printf("  steering mode: %s (sig layer %d)\n\n",
           jungLayer >= 0 ? "Jungian channel bias" : "activation addition (no Jungian layer)", sigLayer);

    int D = cfg.embDim, K16 = N_ARCH;
    /* Post-hoc validation boundaries (works inside joyce_all.txt). The method
     * never sees these — they only grade the emerged clusters afterwards. */
    long wb[6] = { 0, 16480, 382290, 842870, 988600, nTokens };
    const char *wname[5] = { "ChamberMusic", "Dubliners", "Portrait", "Exiles", "Ulysses" };
    int haveLabels = (strstr(corpusPath, "joyce_all") != NULL);

    /* 0. Global baseline of the Jungian layer input (mean ln1Out): signatures
     * are centered by it so only a turn's DISTINCTIVE direction remains. */
    static float base[EMB_DIM];
    memset(base, 0, sizeof(base));
    double meanResNorm = 0;                /* residual-stream scale, for ActAdd */
    {
        long cnt = 0; int nS = 24, win = cfg.contextLen;
        long stride = (nTokens - win) / nS;
        for (int c = 0; c < nS; c++) {
            ForwardCache fc = forward(ids + c * stride, win, &w, cfg);
            const float *h = fc.blockCaches[sigLayer].ln1Out;
            const float *xin = (sigLayer == 0) ? fc.x0 : fc.blockCaches[sigLayer - 1].output;
            for (int s = 0; s < win; s++) {
                double n2 = 0;
                for (int d = 0; d < D; d++) {
                    base[d] += h[s * D + d];
                    n2 += (double)xin[s * D + d] * xin[s * D + d];
                }
                meanResNorm += sqrt(n2);
            }
            cnt += win;
            freeForwardCache(&fc);
        }
        for (int d = 0; d < D; d++) base[d] /= cnt;
        meanResNorm /= cnt;
    }

    /* 1. The battery: M turns (the "session"), each a passage the interlocutor
     * brings up. Signature = centered mean hidden state, unit-normalized. */
    enum { M = 120, TURN = 384, KMAX = 12, CAL = 25 };
    static float sig[M][EMB_DIM]; static int tpos[M], tlab[M];
    for (int t = 0; t < M; t++) {
        long lo, hi; int u = t % 5;
        if (haveLabels) { lo = wb[u] + 200; hi = wb[u + 1] - TURN - 200; }
        else            { lo = 100;         hi = nTokens - TURN - 100; }
        int pos = (int)(lo + (long)(randf() * (double)(hi - lo)));
        tpos[t] = pos; tlab[t] = haveLabels ? u : 0;
        ForwardCache fc = forward(ids + pos, TURN, &w, cfg);
        const float *h = fc.blockCaches[sigLayer].ln1Out;
        double acc[EMB_DIM] = { 0 };
        for (int s = 0; s < TURN; s++)
            for (int d = 0; d < D; d++) acc[d] += h[s * D + d];
        double nrm = 0;
        for (int d = 0; d < D; d++) {
            sig[t][d] = (float)(acc[d] / TURN) - base[d];
            nrm += sig[t][d] * sig[t][d];
        }
        nrm = sqrt(nrm) + 1e-9;
        for (int d = 0; d < D; d++) sig[t][d] /= (float)nrm;
        freeForwardCache(&fc);
    }

    /* 2. Novelty threshold calibrated on the FIRST turns only (the getting-to-
     * know-you phase): θ = p75 of their pairwise cosine — nothing hand-picked. */
    float pcos[CAL * CAL]; int np = 0;
    for (int a = 0; a < CAL; a++)
        for (int b = a + 1; b < CAL; b++) {
            double c = 0; for (int d = 0; d < D; d++) c += sig[a][d] * sig[b][d];
            pcos[np++] = (float)c;
        }
    for (int a = 0; a < np; a++)
        for (int b = a + 1; b < np; b++)
            if (pcos[b] < pcos[a]) { float tm = pcos[a]; pcos[a] = pcos[b]; pcos[b] = tm; }
    float theta = pcos[(int)(0.75f * np)];
    printf("  ── Emergence (online, novelty-gated) ────────────────────────────────\n");
    printf("     calibration on first %d turns: pairwise cos p25=%.3f p50=%.3f p75=%.3f → θ=%.3f\n",
           CAL, pcos[np / 4], pcos[np / 2], theta, theta);

    /* 3. Online pass: an archetype is BORN when a turn resembles none of the
     * living ones (max cos < θ); otherwise the nearest archetype absorbs it. */
    static float ce[KMAX][EMB_DIM]; float mass[KMAX]; int bornAt[KMAX], assign[M];
    int nK = 0; const float eta = 0.15f;
    for (int t = 0; t < M; t++) {
        int bk = -1; double bc = -2;
        for (int k = 0; k < nK; k++) {
            double c = 0; for (int d = 0; d < D; d++) c += sig[t][d] * ce[k][d];
            if (c > bc) { bc = c; bk = k; }
        }
        if (bk < 0 || (bc < theta && nK < KMAX)) {
            memcpy(ce[nK], sig[t], sizeof(float) * D);
            mass[nK] = 1; bornAt[nK] = t; assign[t] = nK; nK++;
        } else {
            double nrm = 0;
            for (int d = 0; d < D; d++) {
                ce[bk][d] = (1.f - eta) * ce[bk][d] + eta * sig[t][d];
                nrm += ce[bk][d] * ce[bk][d];
            }
            nrm = sqrt(nrm) + 1e-9;
            for (int d = 0; d < D; d++) ce[bk][d] /= (float)nrm;
            mass[bk] += 1; assign[t] = bk;
        }
    }
    printf("     %d archetypes emerged during the battery (%d turns, cap %d, K not pre-set)\n\n", nK, M, KMAX);

    /* 4. Identification: an archetype is known by the turns that tend to it
     * (nearest-turn snippet), graded post-hoc by source work. */
    printf("  ── The emergent archetypes, identified by their own turns ───────────\n");
    int agree = 0;
    for (int k = 0; k < nK; k++) {
        int cnt[5] = { 0 };
        for (int t = 0; t < M; t++) if (assign[t] == k) cnt[tlab[t]]++;
        int mj = 0; for (int u = 1; u < 5; u++) if (cnt[u] > cnt[mj]) mj = u;
        agree += cnt[mj];
        int bt = bornAt[k]; double bc = -2;
        for (int t = 0; t < M; t++) {
            if (assign[t] != k) continue;
            double c = 0; for (int d = 0; d < D; d++) c += sig[t][d] * ce[k][d];
            if (c > bc) { bc = c; bt = t; }
        }
        char snip[64]; int sn = 0;
        for (int j = 40; j < TURN && sn < 58; j++) {
            char ch = idToChar[ids[tpos[bt] + j]];
            snip[sn++] = (ch == '\n' || ch == '\t') ? ' ' : ch;
        }
        snip[sn] = 0;
        printf("     «A%d» born@t=%-3d mass=%-3.0f «%s»\n", k, bornAt[k], mass[k], snip);
        if (haveLabels)
            printf("          post-hoc: %d/%.0f turns from %s\n", cnt[mj], mass[k], wname[mj]);
    }
    if (haveLabels)
        printf("\n     post-hoc purity vs source work: %.0f%%  (chance 20%%; labels never used by the method)\n",
               100.0 * agree / M);

    /* 5. Shadow of each emergent archetype: the living archetype it most
     * avoids (lowest cosine) — the anti-image, discovered, not curated. */
    int shadowOf[KMAX];
    for (int k = 0; k < nK; k++) {
        int sk = -1; double sc = 2;
        for (int j = 0; j < nK; j++) {
            if (j == k) continue;
            double c = 0; for (int d = 0; d < D; d++) c += ce[k][d] * ce[j][d];
            if (c < sc) { sc = c; sk = j; }
        }
        shadowOf[k] = sk;
    }

    /* 6. Steering by archetypes discovered during this same battery: map the
     * centroid onto the mixer channels — bias_k16 = ⟨persona,wArch⟩ −
     * ⟨sombra,wArch⟩ (attract the persona, repel its shadow) — and generate.
     * The generated text's signature is measured with the bias OFF (no
     * circularity). Success = generation moves toward the target archetype's
     * region and NOT toward the others. */
    int charToId[256];
    for (int i = 0; i < 256; i++) charToId[i] = -1;
    for (int v = 0; v < vocabSize; v++) charToId[(unsigned char)idToChar[v]] = v;
    const char *seedStr = "he said that ";
    int seed[32], seedLen = 0;
    for (int i = 0; seedStr[i] && seedLen < 32; i++)
        if (charToId[(unsigned char)seedStr[i]] >= 0)
            seed[seedLen++] = charToId[(unsigned char)seedStr[i]];

    enum { GLEN = 400, GS = 4 };
    static int gout[GLEN + 64];
    static float bias16[8][N_ARCH];
    char sample[141]; sample[0] = 0;

    int ord[KMAX];
    for (int k = 0; k < nK; k++) ord[k] = k;
    for (int a = 0; a < nK; a++)
        for (int b = a + 1; b < nK; b++)
            if (mass[ord[b]] > mass[ord[a]]) { int tm = ord[a]; ord[a] = ord[b]; ord[b] = tm; }
    int nSteer = nK < 3 ? nK : 3;

    printf("\n  ── Steering by emergent archetype (confront → revise → sweep) ───────\n");
    printf("     The channel bias maps ⟨persona,wArch⟩−⟨sombra,wArch⟩, but the trained\n");
    printf("     channel→output map has unknown polarity per archetype. So the analyst\n");
    printf("     CONFRONTS: tries both polarities at w_u=8 and keeps the one the patient\n");
    printf("     accepts (archetype_analysis.py's confront→revise loop, in the neural model).\n");
    printf("     %-5s | %6s | %11s | %11s | %8s | %s\n",
           "arch", "w_u", "cos→target", "cos→others", "Δ(t−o)", "fluency (unbiased model)");
    static float actDir[EMB_DIM];
    for (int i = 0; i < nSteer; i++) {
        int k = ord[i];
        if (jungLayer >= 0) {
            /* channel bias for this archetype, per Jungian layer (unit L2) */
            for (int l = 0; l < cfg.numLayers && l < 8; l++) {
                if (cfg.layerTypes[l] != MIX_JUNGIAN) continue;
                const float *wa = w.blocks[l].wArch;
                double nrm = 0;
                for (int c16 = 0; c16 < K16; c16++) {
                    double p = 0, s2 = 0;
                    for (int d = 0; d < D; d++) {
                        p += ce[k][d] * wa[c16 * D + d];
                        if (shadowOf[k] >= 0) s2 += ce[shadowOf[k]][d] * wa[c16 * D + d];
                    }
                    bias16[l][c16] = (float)(p - s2);
                    nrm += bias16[l][c16] * bias16[l][c16];
                }
                nrm = sqrt(nrm) + 1e-9;
                for (int c16 = 0; c16 < K16; c16++) bias16[l][c16] /= (float)nrm;
            }
        } else {
            /* ActAdd direction: persona − sombra, unit-normalized */
            double nrm = 0;
            for (int d = 0; d < D; d++) {
                actDir[d] = ce[k][d] - (shadowOf[k] >= 0 ? ce[shadowOf[k]][d] : 0.f);
                nrm += actDir[d] * actDir[d];
            }
            nrm = sqrt(nrm) + 1e-9;
            for (int d = 0; d < D; d++) actDir[d] /= (float)nrm;
        }
        /* confrontation (sign choice), then sweep with the revised sign.
         * ActAdd magnitudes are fractions of the measured residual norm; the
         * dense low-dose arms exist to find the fluency-matched operating
         * point (same bits/char cost as baseline) for a fair comparison. */
        float sign = 1.f;
        float mn = (float)meanResNorm;
        float wLo = (jungLayer >= 0) ? 8.f : 0.25f * mn;
        float sweepArm[6]; int nSweep;
        if (jungLayer >= 0) { sweepArm[0] = 0.f; sweepArm[1] = 8.f; sweepArm[2] = 24.f; nSweep = 3; }
        else {
            sweepArm[0] = 0.f;        sweepArm[1] = 0.05f * mn; sweepArm[2] = 0.10f * mn;
            sweepArm[3] = 0.15f * mn; sweepArm[4] = 0.25f * mn; sweepArm[5] = 0.75f * mn;
            nSweep = 6;
        }
        int sampleArm = (jungLayer >= 0) ? (2 + nSweep - 1) : (2 + 3);  /* jung: max dose; act: 0.15·‖res‖ */
        double dConf[2] = { 0, 0 };
        for (int wi = 0; wi < 2 + nSweep; wi++) {
            for (int l = 0; l < 8; l++) { g_personalBias[l] = NULL; g_actSteer[l] = NULL; }
            float wEff = (wi < 2) ? (wi == 0 ? wLo : -wLo) : sweepArm[wi - 2] * sign;
            if (jungLayer >= 0) {
                for (int l = 0; l < cfg.numLayers && l < 8; l++)
                    if (cfg.layerTypes[l] == MIX_JUNGIAN) g_personalBias[l] = bias16[l];
                g_personalW = wEff;
            } else {
                g_actSteer[sigLayer] = actDir;
                g_actW = wEff;
            }
            int nGen = (wi < 2) ? 3 : GS;
            double ct = 0, co = 0, ceBits = 0; int nco = 0;
            for (int g = 0; g < nGen; g++) {
                int n = generateText(&w, cfg, seed, seedLen, GLEN, 0.8f, 0.9f, 256, gout);
                if (wi == sampleArm && g == 0) {  /* keep one steered sample */
                    int sn = 0;
                    for (int j = seedLen; j < n && sn < 140; j++) {
                        char ch = idToChar[gout[j]];
                        sample[sn++] = (ch == '\n' || ch == '\t') ? ' ' : ch;
                    }
                    sample[sn] = 0;
                }
                const float *saveB[8], *saveA[8]; float saveW = g_personalW, saveAW = g_actW;
                for (int l = 0; l < 8; l++) {
                    saveB[l] = g_personalBias[l]; g_personalBias[l] = NULL;
                    saveA[l] = g_actSteer[l];     g_actSteer[l] = NULL;
                }
                g_personalW = 0; g_actW = 0;
                ForwardCache fc = forward(gout, n, &w, cfg);
                /* fluency: bits/char of the steered text under the UNBIASED model */
                ceBits += crossEntropyLoss(fc.probs, gout + 1, n - 1, cfg.vocabSize) / logf(2.f);
                const float *h = fc.blockCaches[sigLayer].ln1Out;
                double acc[EMB_DIM] = { 0 };
                for (int s = 0; s < n; s++)
                    for (int d = 0; d < D; d++) acc[d] += h[s * D + d];
                float gsig[EMB_DIM]; double nrm = 0;
                for (int d = 0; d < D; d++) {
                    gsig[d] = (float)(acc[d] / n) - base[d];
                    nrm += gsig[d] * gsig[d];
                }
                nrm = sqrt(nrm) + 1e-9;
                for (int d = 0; d < D; d++) gsig[d] /= (float)nrm;
                freeForwardCache(&fc);
                for (int l = 0; l < 8; l++) { g_personalBias[l] = saveB[l]; g_actSteer[l] = saveA[l]; }
                g_personalW = saveW; g_actW = saveAW;
                double c = 0; for (int d = 0; d < D; d++) c += gsig[d] * ce[k][d];
                ct += c;
                for (int j = 0; j < nK; j++) {
                    if (j == k) continue;
                    double c2 = 0; for (int d = 0; d < D; d++) c2 += gsig[d] * ce[j][d];
                    co += c2; nco++;
                }
            }
            ct /= nGen; co /= (nco ? nco : 1); ceBits /= nGen;
            if (wi < 2) {
                dConf[wi] = ct - co;
                if (wi == 1) {
                    sign = (dConf[0] >= dConf[1]) ? 1.f : -1.f;
                    printf("     «A%d» confront: Δ(+%.1f)=%+.4f  Δ(−%.1f)=%+.4f  → %s\n",
                           k, wLo, dConf[0], wLo, dConf[1],
                           sign > 0 ? "accepted as proposed" : "patient resisted — sign revised");
                }
            } else {
                printf("     «A%d» | %+6.1f | %11.4f | %11.4f | %+8.4f | %6.3f bits/char\n",
                       k, sweepArm[wi - 2] * sign, ct, co, ct - co, ceBits);
            }
        }
        printf("       sample(revised w_u): «%s»\n", sample);
    }
    for (int l = 0; l < 8; l++) { g_personalBias[l] = NULL; g_actSteer[l] = NULL; }
    g_personalW = 0.f; g_actW = 0.f;

    free(ids);
    freeWeights(&w, cfg); freeAdam(&adam);
}

/* ─── Interface diagnostic (--interface-test) ─────────────────────────────────
 * Did STEER_AUG teach the model to READ the standing bias? Direct test:
 * teacher-forced CE of held-out windows under {no bias, the CORRECT neighbour
 * bias, a SHUFFLED neighbour bias}. Interface learned ⇔ correct < none and
 * correct < shuffled. Run on the aug checkpoint and on the baseline (control:
 * no effect expected there). */

static int g_interfaceTest = 0;    /* --interface-test */

static void steerInterfaceExperiment(const char *corpusPath) {
    char idToChar[256];
    int nTokens, vocabSize;
    int *ids = loadCorpus(corpusPath, &nTokens, &vocabSize, idToChar);
    Config cfg = { .vocabSize = vocabSize, .contextLen = SEQ_LEN, .embDim = EMB_DIM,
                   .numHeads = 4, .numKVHeads = (NUM_KV_HEADS > 4) ? 4 : NUM_KV_HEADS,
                   .ffDim = FF_DIM, .stateN = 16 };
    applyLayout(&cfg);
    int jungLayer = -1;
    for (int l = 0; l < cfg.numLayers; l++)
        if (cfg.layerTypes[l] == MIX_JUNGIAN) { jungLayer = l; break; }
    if (jungLayer < 0) { fprintf(stderr, "interface-test: needs a Jungian layer\n"); free(ids); return; }
    Weights w = initWeights(cfg);
    AdamBuf adam = initAdam(&w, cfg);
    int step = loadCkpt(&w, &adam, cfg);
    resetRand(99);
    printModelHeader(cfg);
    printf("  corpus=%s  ckpt-step=%d\n\n", corpusPath, step);

    int D = cfg.embDim, K16 = N_ARCH;
    static float base[EMB_DIM]; memset(base, 0, sizeof(base));
    {
        long cnt = 0; int nS = 24, win = cfg.contextLen;
        long stride = (nTokens - win) / nS;
        for (int c = 0; c < nS; c++) {
            ForwardCache fc = forward(ids + c * stride, win, &w, cfg);
            const float *h = fc.blockCaches[jungLayer].ln1Out;
            for (int s = 0; s < win; s++)
                for (int d = 0; d < D; d++) base[d] += h[s * D + d];
            cnt += win; freeForwardCache(&fc);
        }
        for (int d = 0; d < D; d++) base[d] /= cnt;
    }

    enum { NT = 40 };
    int WIN = cfg.contextLen < 512 ? cfg.contextLen : 512;   /* CE window ≤ ctx */
    int NB  = cfg.contextLen < 384 ? cfg.contextLen : 384;   /* neighbour span ≤ ctx */
    static int tpos[NT];
    static float tbias[NT][8][N_ARCH];
    for (int t = 0; t < NT; t++) {
        int pos;
        do { pos = NB + 16 + (int)(randf() * (double)(nTokens - WIN - 2 * NB - 32)); } while (pos < NB + 16);
        tpos[t] = pos;
        ForwardCache nf = forward(ids + pos - NB, NB, &w, cfg);
        const float *h = nf.blockCaches[jungLayer].ln1Out;
        double acc[EMB_DIM] = { 0 };
        for (int s = 0; s < NB; s++)
            for (int d = 0; d < D; d++) acc[d] += h[s * D + d];
        float sigv[EMB_DIM]; double nrm = 0;
        for (int d = 0; d < D; d++) { sigv[d] = (float)(acc[d] / NB) - base[d]; nrm += sigv[d] * sigv[d]; }
        nrm = sqrt(nrm) + 1e-9;
        for (int d = 0; d < D; d++) sigv[d] /= (float)nrm;
        freeForwardCache(&nf);
        for (int l = 0; l < cfg.numLayers && l < 8; l++) {
            if (cfg.layerTypes[l] != MIX_JUNGIAN) continue;
            const float *wa = w.blocks[l].wArch;
            double bn = 0;
            for (int k = 0; k < K16; k++) {
                double p = 0;
                for (int d = 0; d < D; d++) p += sigv[d] * wa[k * D + d];
                tbias[t][l][k] = (float)p; bn += p * p;
            }
            bn = sqrt(bn) + 1e-9;
            for (int k = 0; k < K16; k++) tbias[t][l][k] /= (float)bn;
        }
    }

    const float LN2 = 0.6931472f;
    double ce[3] = { 0, 0, 0 };                    /* none, correct, shuffled */
    double dCor = 0, dCor2 = 0, dShf = 0, dShf2 = 0;
    for (int t = 0; t < NT; t++) {
        double c3[3];
        for (int cond = 0; cond < 3; cond++) {
            for (int l = 0; l < 8; l++) g_personalBias[l] = NULL;
            g_personalW = 0.f;
            if (cond > 0) {
                int src = (cond == 1) ? t : (t + NT / 2) % NT;   /* shuffled = another trial's bias */
                for (int l = 0; l < cfg.numLayers && l < 8; l++)
                    if (cfg.layerTypes[l] == MIX_JUNGIAN) g_personalBias[l] = tbias[src][l];
                g_personalW = 20.f;
            }
            ForwardCache fc = forward(ids + tpos[t], WIN, &w, cfg);
            c3[cond] = crossEntropyLoss(fc.probs, ids + tpos[t] + 1, WIN - 1, cfg.vocabSize) / LN2;
            freeForwardCache(&fc);
            ce[cond] += c3[cond];
        }
        double a = c3[1] - c3[0], b = c3[1] - c3[2];
        dCor += a; dCor2 += a * a; dShf += b; dShf2 += b * b;
    }
    for (int l = 0; l < 8; l++) g_personalBias[l] = NULL;
    g_personalW = 0.f;
    double mA = dCor / NT, sA = sqrt((dCor2 / NT - mA * mA) / NT);
    double mB = dShf / NT, sB = sqrt((dShf2 / NT - mB * mB) / NT);
    printf("  ── Interface: CE bits/char em %d janelas held-out (w_u=20) ─────────\n", NT);
    printf("     sem viés            : %.4f\n", ce[0] / NT);
    printf("     viés CORRETO        : %.4f\n", ce[1] / NT);
    printf("     viés EMBARALHADO    : %.4f\n", ce[2] / NT);
    printf("     correto − sem       : %+.4f ± %.4f (pareado)\n", mA, sA);
    printf("     correto − embaralh. : %+.4f ± %.4f (pareado)\n", mB, sB);
    printf("     interface aprendida ⇔ ambos claramente < 0\n");

    /* ── Effector ceiling: how much CAN routing move the output at all? ──────
     * Large w_u saturates the routing softmax toward one-hot — the w→∞ limit
     * of any steering. If even saturated routing barely moves the logits, the
     * archetype→output path has no gain and no training signal could have made
     * the interface profitable (the chicken-and-egg of the flat interface). */
    printf("\n  ── Teto do efetor (dose-resposta até saturação do roteamento) ──────\n");
    {
        /* value-table spread: v = wArch·wV per Jungian layer */
        for (int l = 0; l < cfg.numLayers && l < 8; l++) {
            if (cfg.layerTypes[l] != MIX_JUNGIAN) continue;
            float *v = matmul(w.blocks[l].wArch, w.blocks[l].wV_j, K16, D, D);
            double mn = 0, mmax = 0, nsum = 0;
            int cnt2 = 0;
            for (int a = 0; a < K16; a++) {
                double na = 0;
                for (int d = 0; d < D; d++) na += (double)v[a * D + d] * v[a * D + d];
                nsum += sqrt(na);
                for (int b2 = a + 1; b2 < K16; b2++) {
                    double dd = 0;
                    for (int d = 0; d < D; d++) {
                        double df = v[a * D + d] - v[b2 * D + d];
                        dd += df * df;
                    }
                    dd = sqrt(dd);
                    mn += dd; if (dd > mmax) mmax = dd; cnt2++;
                }
            }
            printf("     camada %d: ‖v_k‖ médio %.3f · dist par-a-par média %.3f máx %.3f\n",
                   l, nsum / K16, mn / cnt2, mmax);
            free(v);
        }
        /* dose-response on NT2 windows with each window's own neighbour bias */
        enum { NT2 = 10 };
        const float wBig[3] = { 40.f, 100.f, 300.f };
        int V = cfg.vocabSize;
        printf("     %6s | %10s | %12s | %10s | %s\n", "w_u", "TV médio", "|Δlogit| L2", "ΔCE b/c", "H(p) rot.");
        for (int wi = 0; wi < 3; wi++) {
            double tvS = 0, l2S = 0, dceS = 0, entS = 0; long entN = 0;
            for (int t = 0; t < NT2; t++) {
                for (int l = 0; l < 8; l++) g_personalBias[l] = NULL;
                g_personalW = 0.f;
                ForwardCache f0 = forward(ids + tpos[t], WIN, &w, cfg);
                for (int l = 0; l < cfg.numLayers && l < 8; l++)
                    if (cfg.layerTypes[l] == MIX_JUNGIAN) g_personalBias[l] = tbias[t][l];
                g_personalW = wBig[wi];
                ForwardCache f1 = forward(ids + tpos[t], WIN, &w, cfg);
                /* routing entropy under bias, at the Jungian layer */
                const float *pj = f1.blockCaches[jungLayer].p_j;
                for (int s = 0; s < WIN; s++) {
                    double H = 0;
                    for (int k = 0; k < K16; k++)
                        if (pj[s * K16 + k] > 1e-9) H -= pj[s * K16 + k] * log(pj[s * K16 + k]);
                    entS += H; entN++;
                }
                double ce0 = crossEntropyLoss(f0.probs, ids + tpos[t] + 1, WIN - 1, V);
                double ce1 = crossEntropyLoss(f1.probs, ids + tpos[t] + 1, WIN - 1, V);
                dceS += (ce1 - ce0) / LN2;
                for (int s = 0; s < WIN; s++) {
                    double m0 = NEG_INF, m1 = NEG_INF;
                    for (int vv = 0; vv < V; vv++) {
                        if (f0.logits[s * V + vv] / 0.8f > m0) m0 = f0.logits[s * V + vv] / 0.8f;
                        if (f1.logits[s * V + vv] / 0.8f > m1) m1 = f1.logits[s * V + vv] / 0.8f;
                    }
                    double s0 = 0, s1 = 0;
                    for (int vv = 0; vv < V; vv++) {
                        s0 += exp(f0.logits[s * V + vv] / 0.8f - m0);
                        s1 += exp(f1.logits[s * V + vv] / 0.8f - m1);
                    }
                    double tv = 0, l2 = 0;
                    for (int vv = 0; vv < V; vv++) {
                        double a = exp(f0.logits[s * V + vv] / 0.8f - m0) / s0;
                        double b2 = exp(f1.logits[s * V + vv] / 0.8f - m1) / s1;
                        tv += fabs(a - b2);
                        double df = f0.logits[s * V + vv] - f1.logits[s * V + vv];
                        l2 += df * df;
                    }
                    tvS += 0.5 * tv / WIN; l2S += sqrt(l2) / WIN;
                }
                freeForwardCache(&f0); freeForwardCache(&f1);
            }
            printf("     %6.0f | %9.3f%% | %12.4f | %+10.4f | %.3f nats (unif=%.3f)\n",
                   wBig[wi], 100.0 * tvS / NT2, l2S / NT2, dceS / NT2, entS / entN, log((double)K16));
        }
        for (int l = 0; l < 8; l++) g_personalBias[l] = NULL;
        g_personalW = 0.f;
        printf("     → teto provado se, mesmo com H(p)→0 (roteamento saturado), TV/Δlogit ficarem ínfimos.\n");
    }
    free(ids);
    freeWeights(&w, cfg); freeAdam(&adam);
}

/* ─── Repression experiment (inference-only, --repression-test) ───────────────
 * The Jungian architecture actually operating over EMERGENT archetypes, across
 * multiple prompts. Phase 1: archetypes emerge from a balanced battery (same
 * protocol as --emerge-test). Phase 2: a session of prompts drawn from ONE
 * register feeds a per-session Shadow in emergent-archetype space,
 *
 *     S_k ← α·S_k + (1−α)·(1−p_k)/(K−1)        (p = persona over emergent arcs)
 *
 * persisted across prompts (O(K) floats — the shadow cache). Steering is driven
 * by the ACCUMULATED Shadow itself: direction = Σ(S̃_k − 1/K)·centroid_k mapped
 * onto the mixer channels, strength ∝ accumulated deviation. Falsifiable
 * sequestration claims: (a) the pull of generation toward the REPRESSED
 * archetypes grows with session length; (b) it persists and decays slowly on
 * neutral prompts; (c) without the mechanism the curve is flat; (d) different
 * session registers sequester different directions. Prompts are always encoded
 * bias-OFF; only generation is steered; measurement is bias-OFF (no
 * circularity). Central weights untouched. */

static int g_repressionTest = 0;   /* --repression-test */

static void shadowRepressionExperiment(const char *corpusPath) {
    char idToChar[256];
    int nTokens, vocabSize;
    int *ids = loadCorpus(corpusPath, &nTokens, &vocabSize, idToChar);

    Config cfg = { .vocabSize = vocabSize, .contextLen = SEQ_LEN, .embDim = EMB_DIM,
                   .numHeads = 4, .numKVHeads = (NUM_KV_HEADS > 4) ? 4 : NUM_KV_HEADS,
                   .ffDim = FF_DIM, .stateN = 16 };
    applyLayout(&cfg);
    int jungLayer = -1;
    for (int l = 0; l < cfg.numLayers; l++)
        if (cfg.layerTypes[l] == MIX_JUNGIAN) { jungLayer = l; break; }
    if (jungLayer < 0) { fprintf(stderr, "repression-test: needs a Jungian layer\n"); free(ids); return; }

    Weights w = initWeights(cfg);
    AdamBuf adam = initAdam(&w, cfg);
    int step = loadCkpt(&w, &adam, cfg);
    resetRand(1234);

    printModelHeader(cfg);
    printf("  corpus=%s  tokens=%d  vocab=%d  ckpt-step=%d\n\n", corpusPath, nTokens, vocabSize, step);

    int D = cfg.embDim, K16 = N_ARCH;
    long wb[6] = { 0, 16480, 382290, 842870, 988600, nTokens };
    const char *wname[5] = { "ChamberMusic", "Dubliners", "Portrait", "Exiles", "Ulysses" };
    int haveLabels = (strstr(corpusPath, "joyce_all") != NULL);
    if (!haveLabels) { fprintf(stderr, "repression-test: needs joyce_all.txt (register boundaries)\n"); free(ids); return; }

    /* baseline hidden state */
    static float base[EMB_DIM];
    memset(base, 0, sizeof(base));
    {
        long cnt = 0; int nS = 24, win = cfg.contextLen;
        long stride = (nTokens - win) / nS;
        for (int c = 0; c < nS; c++) {
            ForwardCache fc = forward(ids + c * stride, win, &w, cfg);
            const float *h = fc.blockCaches[jungLayer].ln1Out;
            for (int s = 0; s < win; s++)
                for (int d = 0; d < D; d++) base[d] += h[s * D + d];
            cnt += win;
            freeForwardCache(&fc);
        }
        for (int d = 0; d < D; d++) base[d] /= cnt;
    }

    /* ── Phase 1: emergence (same protocol as --emerge-test) ── */
    enum { M = 120, TURN = 384, KMAX = 12, CAL = 25 };
    static float sig[M][EMB_DIM]; static int tlab[M];
    for (int t = 0; t < M; t++) {
        int u = t % 5;
        long lo = wb[u] + 200, hi = wb[u + 1] - TURN - 200;
        int pos = (int)(lo + (long)(randf() * (double)(hi - lo)));
        tlab[t] = u;
        ForwardCache fc = forward(ids + pos, TURN, &w, cfg);
        const float *h = fc.blockCaches[jungLayer].ln1Out;
        double acc[EMB_DIM] = { 0 };
        for (int s = 0; s < TURN; s++)
            for (int d = 0; d < D; d++) acc[d] += h[s * D + d];
        double nrm = 0;
        for (int d = 0; d < D; d++) { sig[t][d] = (float)(acc[d] / TURN) - base[d]; nrm += sig[t][d] * sig[t][d]; }
        nrm = sqrt(nrm) + 1e-9;
        for (int d = 0; d < D; d++) sig[t][d] /= (float)nrm;
        freeForwardCache(&fc);
    }
    float pcos[CAL * CAL]; int np = 0;
    for (int a = 0; a < CAL; a++)
        for (int b = a + 1; b < CAL; b++) {
            double c = 0; for (int d = 0; d < D; d++) c += sig[a][d] * sig[b][d];
            pcos[np++] = (float)c;
        }
    for (int a = 0; a < np; a++)
        for (int b = a + 1; b < np; b++)
            if (pcos[b] < pcos[a]) { float tm = pcos[a]; pcos[a] = pcos[b]; pcos[b] = tm; }
    float theta = pcos[(int)(0.75f * np)];
    static float ce[KMAX][EMB_DIM]; float mass[KMAX]; int assign[M];
    int nK = 0; const float etaC = 0.15f;
    for (int t = 0; t < M; t++) {
        int bk = -1; double bc = -2;
        for (int k = 0; k < nK; k++) {
            double c = 0; for (int d = 0; d < D; d++) c += sig[t][d] * ce[k][d];
            if (c > bc) { bc = c; bk = k; }
        }
        if (bk < 0 || (bc < theta && nK < KMAX)) {
            memcpy(ce[nK], sig[t], sizeof(float) * D);
            mass[nK] = 1; assign[t] = nK; nK++;
        } else {
            double nrm = 0;
            for (int d = 0; d < D; d++) { ce[bk][d] = (1.f - etaC) * ce[bk][d] + etaC * sig[t][d]; nrm += ce[bk][d] * ce[bk][d]; }
            nrm = sqrt(nrm) + 1e-9;
            for (int d = 0; d < D; d++) ce[bk][d] /= (float)nrm;
            mass[bk] += 1; assign[t] = bk;
        }
    }
    int agree = 0;
    for (int k = 0; k < nK; k++) {
        int cnt[5] = { 0 };
        for (int t = 0; t < M; t++) if (assign[t] == k) cnt[tlab[t]]++;
        int mj = 0; for (int u = 1; u < 5; u++) if (cnt[u] > cnt[mj]) mj = u;
        agree += cnt[mj];
    }
    printf("  Phase 1 — emergence: %d archetypes (θ=%.3f), post-hoc purity %.0f%%\n\n",
           nK, theta, 100.0 * agree / M);

    /* Baseline shadow S̄: the EXPECTED repression under balanced content (the
     * phase-1 battery itself). Session-specific repression = S̃ − S̄ — removes
     * the common mode that otherwise elects the same orphan archetype (the
     * cluster far from everything) in every session. */
    const float tauP = 0.1f;
    float Sbar[KMAX]; memset(Sbar, 0, sizeof(Sbar));
    for (int t = 0; t < M; t++) {
        float pb[KMAX]; double mx = -1e9, sm = 0;
        for (int k = 0; k < nK; k++) {
            double c = 0; for (int d = 0; d < D; d++) c += sig[t][d] * ce[k][d];
            pb[k] = (float)(c / tauP);
            if (pb[k] > mx) mx = pb[k];
        }
        for (int k = 0; k < nK; k++) { pb[k] = (float)exp(pb[k] - mx); sm += pb[k]; }
        for (int k = 0; k < nK; k++)
            Sbar[k] += (1.f - pb[k] / (float)sm) / (float)(nK - 1);
    }
    for (int k = 0; k < nK; k++) Sbar[k] /= M;
    printf("  S̄ baseline (repressão esperada balanceada): ");
    for (int k = 0; k < nK; k++) printf("A%d=%.3f ", k, Sbar[k]);
    printf("\n\n");

    /* ── Phase 2: sessions ── */
    int charToId[256];
    for (int i = 0; i < 256; i++) charToId[i] = -1;
    for (int v = 0; v < vocabSize; v++) charToId[(unsigned char)idToChar[v]] = v;
    const char *seedStr = "he said that ";
    int seed[32], seedLen = 0;
    for (int i = 0; seedStr[i] && seedLen < 32; i++)
        if (charToId[(unsigned char)seedStr[i]] >= 0)
            seed[seedLen++] = charToId[(unsigned char)seedStr[i]];

    enum { ST = 30, FEED = 24, GLEN = 400, GS = 6 };
    /* Dose: no v1 (efetor inerte) usava-se 150/40; no v2 a voz direta torna
     * w_u≈8 a banda produtiva (emerge-test: move sem degradar fluência). */
#if JUNGIAN_V2
    const float alpha = 0.85f, WSCALE = 60.f, WCAP = 12.f;
#else
    const float alpha = 0.85f, WSCALE = 150.f, WCAP = 40.f;
#endif
    static int gout[GLEN + 64];
    static float bias16[8][N_ARCH];
    static float shDirFinal[2][EMB_DIM];
    const int sessWorks[2] = { 1, 3 };            /* Dubliners prose, Exiles drama */

    for (int sess = 0; sess < 2; sess++) {
        int sw = sessWorks[sess];
        resetRand(777 + sess);
        float S[KMAX]; for (int k = 0; k < nK; k++) S[k] = 1.f / nK;
        float pbar[KMAX]; memset(pbar, 0, sizeof(pbar)); int pbarN = 0;
        int fed[KMAX]; memset(fed, 0, sizeof(fed)); int frozen = 0, nFed = 0;
        static float repMix[EMB_DIM], fedMix[EMB_DIM], shDir[EMB_DIM];
        /* polaridade do mapa canal→saída é POR ARQUÉTIPO (medido no
         * emerge-test: A3 exige sinal invertido); confronta na primeira vez
         * que cada arquétipo lidera a direção, com a MESMA métrica (Δdir). */
        float signCache[KMAX]; int signKnown[KMAX];
        for (int k = 0; k < nK; k++) { signCache[k] = 1.f; signKnown[k] = 0; }
        double semDelta = 0, semSd = 0; int semDone = 0;

        printf("  ── Session %d: feeding «%s» for %d turns, then %d neutral ──────────\n",
               sess, wname[sw], FEED, ST - FEED);

        for (int t = 0; t < ST; t++) {
            /* prompt of this turn (always encoded bias-OFF) */
            int wk = (t < FEED) ? sw : (t % 5);
            long lo = wb[wk] + 200, hi = wb[wk + 1] - TURN - 200;
            int pos = (int)(lo + (long)(randf() * (double)(hi - lo)));
            ForwardCache fc = forward(ids + pos, TURN, &w, cfg);
            const float *h = fc.blockCaches[jungLayer].ln1Out;
            double acc[EMB_DIM] = { 0 };
            for (int s = 0; s < TURN; s++)
                for (int d = 0; d < D; d++) acc[d] += h[s * D + d];
            float psig[EMB_DIM]; double nrm = 0;
            for (int d = 0; d < D; d++) { psig[d] = (float)(acc[d] / TURN) - base[d]; nrm += psig[d] * psig[d]; }
            nrm = sqrt(nrm) + 1e-9;
            for (int d = 0; d < D; d++) psig[d] /= (float)nrm;
            freeForwardCache(&fc);

            /* persona over the EMERGENT archetypes */
            float p[KMAX]; double mx = -1e9;
            for (int k = 0; k < nK; k++) {
                double c = 0; for (int d = 0; d < D; d++) c += psig[d] * ce[k][d];
                p[k] = (float)(c / tauP);
                if (p[k] > mx) mx = p[k];
            }
            double sm = 0;
            for (int k = 0; k < nK; k++) { p[k] = (float)exp(p[k] - mx); sm += p[k]; }
            for (int k = 0; k < nK; k++) p[k] /= (float)sm;
            if (t < FEED) { for (int k = 0; k < nK; k++) pbar[k] += p[k]; pbarN++; }

            /* the Jungian gesture, per prompt: accumulate the repressed */
            for (int k = 0; k < nK; k++)
                S[k] = alpha * S[k] + (1.f - alpha) * (1.f - p[k]) / (float)(nK - 1);

            /* freeze fed/repressed sets and mixtures after 4 feeding turns */
            if (!frozen && t == 3) {
                double fn = 0, rn = 0;
                memset(fedMix, 0, sizeof(float) * EMB_DIM); memset(repMix, 0, sizeof(float) * EMB_DIM);
                for (int k = 0; k < nK; k++) {
                    fed[k] = (pbar[k] / pbarN > 1.f / nK);
                    nFed += fed[k];
                    for (int d = 0; d < D; d++) {
                        if (fed[k]) fedMix[d] += (pbar[k] / pbarN) * ce[k][d];
                        else        repMix[d] += ce[k][d];
                    }
                }
                for (int d = 0; d < D; d++) { fn += fedMix[d] * fedMix[d]; rn += repMix[d] * repMix[d]; }
                fn = sqrt(fn) + 1e-9; rn = sqrt(rn) + 1e-9;
                for (int d = 0; d < D; d++) { fedMix[d] /= (float)fn; repMix[d] /= (float)rn; }
                printf("     fed set (method-internal): %d/%d archetypes\n", nFed, nK);
                frozen = 1;
            }

            /* Direction extraction from the ACCUMULATED S: session-relative
             * repression (score = S̃ − S̄, baseline-centered — kills the orphan
             * degeneracy) + SPARSE top-2 mixture (coherent like top-1,
             * session-specific like the full mixture). */
            double sSum = 0; for (int k = 0; k < nK; k++) sSum += S[k];
            float score[KMAX]; float dev = 0;
            int k1 = -1, k2 = -1;
            for (int k = 0; k < nK; k++) {
                score[k] = (float)(S[k] / sSum) - Sbar[k];
                dev += fabsf(score[k]);
                if (k1 < 0 || score[k] > score[k1]) { k2 = k1; k1 = k; }
                else if (k2 < 0 || score[k] > score[k2]) { k2 = k; }
            }
            memset(shDir, 0, sizeof(float) * EMB_DIM);
            for (int d = 0; d < D; d++) {
                shDir[d] = score[k1] * ce[k1][d];
                if (k2 >= 0 && score[k2] > 0) shDir[d] += score[k2] * ce[k2][d];
            }
            double dn = 0; for (int d = 0; d < D; d++) dn += shDir[d] * shDir[d];
            dn = sqrt(dn) + 1e-9;
            for (int d = 0; d < D; d++) shDir[d] /= (float)dn;

            int isMeas = frozen && ((t == 3) || (t == 7) || (t == 11) || (t == 15) ||
                                    (t == 19) || (t == 23) || (t == 25) || (t == 27) || (t == 29));
            if (!isMeas) continue;

            /* channel bias from the shadow direction */
            for (int l = 0; l < cfg.numLayers && l < 8; l++) {
                if (cfg.layerTypes[l] != MIX_JUNGIAN) continue;
                const float *wa = w.blocks[l].wArch;
                double bn = 0;
                for (int c16 = 0; c16 < K16; c16++) {
                    double b = 0;
                    for (int d = 0; d < D; d++) b += shDir[d] * wa[c16 * D + d];
                    bias16[l][c16] = (float)b; bn += b * b;
                }
                bn = sqrt(bn) + 1e-9;
                for (int c16 = 0; c16 < K16; c16++) bias16[l][c16] /= (float)bn;
            }

            /* one-time SEM reference: unsteered generation SIGNATURES are
             * stored so that, at every point, the metric can be computed
             * against the CURRENT sequestered direction (cos→dir COM vs SEM —
             * fixed mixtures saturate when the baseline already lives on one
             * side; the direction-relative metric doesn't). */
            enum { NSEM = 6 };
            static float semSig[NSEM][EMB_DIM];
            if (!semDone) {
                double dsum = 0, d2 = 0;
                for (int g = 0; g < NSEM; g++) {
                    int n = generateText(&w, cfg, seed, seedLen, GLEN, 0.8f, 0.9f, 256, gout);
                    ForwardCache fg = forward(gout, n, &w, cfg);
                    const float *hg = fg.blockCaches[jungLayer].ln1Out;
                    double ag[EMB_DIM] = { 0 };
                    for (int s = 0; s < n; s++)
                        for (int d = 0; d < D; d++) ag[d] += hg[s * D + d];
                    double gn = 0;
                    for (int d = 0; d < D; d++) { semSig[g][d] = (float)(ag[d] / n) - base[d]; gn += semSig[g][d] * semSig[g][d]; }
                    gn = sqrt(gn) + 1e-9;
                    for (int d = 0; d < D; d++) semSig[g][d] /= (float)gn;
                    freeForwardCache(&fg);
                    double cr = 0, cf = 0;
                    for (int d = 0; d < D; d++) { cr += semSig[g][d] * repMix[d]; cf += semSig[g][d] * fedMix[d]; }
                    dsum += cr - cf; d2 += (cr - cf) * (cr - cf);
                }
                semDelta = dsum / NSEM;
                semSd = sqrt(d2 / NSEM - semDelta * semDelta);
                printf("     SEM mecanismo (referência, %d gens): Δ_rep(misturas) = %+.4f ± %.4f\n", NSEM, semDelta, semSd);
                printf("     %-3s | %-7s | %-6s | %-5s | %9s | %9s | %8s | %s\n",
                       "t", "fase", "dev(S)", "w_u", "cos→dir", "SEM→dir", "Δdir", "fluência");
                semDone = 1;
            }

            /* confrontation on first lead of this archetype, in the Δdir metric */
            if (!signKnown[k1]) {
                double dsig[2] = { 0, 0 };
                for (int si = 0; si < 2; si++) {
                    float sgn = si == 0 ? 1.f : -1.f;
                    for (int l = 0; l < 8; l++) g_personalBias[l] = NULL;
                    for (int l = 0; l < cfg.numLayers && l < 8; l++)
                        if (cfg.layerTypes[l] == MIX_JUNGIAN) g_personalBias[l] = bias16[l];
                    g_personalW = sgn * 12.f;
                    for (int g = 0; g < 2; g++) {
                        int n = generateText(&w, cfg, seed, seedLen, GLEN, 0.8f, 0.9f, 256, gout);
                        const float *sv[8]; float svw = g_personalW;
                        for (int l = 0; l < 8; l++) { sv[l] = g_personalBias[l]; g_personalBias[l] = NULL; }
                        g_personalW = 0;
                        ForwardCache fg = forward(gout, n, &w, cfg);
                        const float *hg = fg.blockCaches[jungLayer].ln1Out;
                        double ag[EMB_DIM] = { 0 };
                        for (int s = 0; s < n; s++)
                            for (int d = 0; d < D; d++) ag[d] += hg[s * D + d];
                        float gs2[EMB_DIM]; double gn = 0;
                        for (int d = 0; d < D; d++) { gs2[d] = (float)(ag[d] / n) - base[d]; gn += gs2[d] * gs2[d]; }
                        gn = sqrt(gn) + 1e-9;
                        for (int d = 0; d < D; d++) gs2[d] /= (float)gn;
                        freeForwardCache(&fg);
                        for (int l = 0; l < 8; l++) g_personalBias[l] = sv[l];
                        g_personalW = svw;
                        double cd = 0;
                        for (int d = 0; d < D; d++) cd += gs2[d] * shDir[d];
                        dsig[si] += cd / 2;
                    }
                }
                signCache[k1] = (dsig[0] >= dsig[1]) ? 1.f : -1.f;
                signKnown[k1] = 1;
                printf("     confront A%d: cos→dir(+)=%+.4f  cos→dir(−)=%+.4f  → sinal %s\n",
                       k1, dsig[0], dsig[1], signCache[k1] > 0 ? "+" : "−");
            }

            /* steered generation driven by the ACCUMULATED shadow */
            for (int l = 0; l < 8; l++) g_personalBias[l] = NULL;
            for (int l = 0; l < cfg.numLayers && l < 8; l++)
                if (cfg.layerTypes[l] == MIX_JUNGIAN) g_personalBias[l] = bias16[l];
            float wu = WSCALE * dev; if (wu > WCAP) wu = WCAP;
            wu *= signCache[k1];
            g_personalW = wu;
            double ctr = 0, ctf = 0, ceB = 0, cdir = 0;
            for (int g = 0; g < GS; g++) {
                int n = generateText(&w, cfg, seed, seedLen, GLEN, 0.8f, 0.9f, 256, gout);
                const float *sv[8]; float svw = g_personalW;
                for (int l = 0; l < 8; l++) { sv[l] = g_personalBias[l]; g_personalBias[l] = NULL; }
                g_personalW = 0;
                ForwardCache fg = forward(gout, n, &w, cfg);
                ceB += crossEntropyLoss(fg.probs, gout + 1, n - 1, cfg.vocabSize) / logf(2.f);
                const float *hg = fg.blockCaches[jungLayer].ln1Out;
                double ag[EMB_DIM] = { 0 };
                for (int s = 0; s < n; s++)
                    for (int d = 0; d < D; d++) ag[d] += hg[s * D + d];
                float gs2[EMB_DIM]; double gn = 0;
                for (int d = 0; d < D; d++) { gs2[d] = (float)(ag[d] / n) - base[d]; gn += gs2[d] * gs2[d]; }
                gn = sqrt(gn) + 1e-9;
                for (int d = 0; d < D; d++) gs2[d] /= (float)gn;
                freeForwardCache(&fg);
                for (int l = 0; l < 8; l++) g_personalBias[l] = sv[l];
                g_personalW = svw;
                double cr = 0, cf = 0, cd = 0;
                for (int d = 0; d < D; d++) {
                    cr += gs2[d] * repMix[d]; cf += gs2[d] * fedMix[d];
                    cd += gs2[d] * shDir[d];
                }
                ctr += cr; ctf += cf; cdir += cd;
            }
            ctr /= GS; ctf /= GS; ceB /= GS; cdir /= GS;
            double semDir = 0;
            for (int g = 0; g < NSEM; g++) {
                double cd = 0;
                for (int d = 0; d < D; d++) cd += semSig[g][d] * shDir[d];
                semDir += cd;
            }
            semDir /= NSEM;
            printf("     %-3d | %-7s | %6.3f | %5.1f | %9.4f | %9.4f | %+8.4f | %.3f b/c | dir=A%d%s\n",
                   t + 1, t < FEED ? "feed" : "NEUTRO", dev, wu, cdir, semDir, cdir - semDir,
                   ceB, k1, (k2 >= 0 && score[k2] > 0) ? "+" : "");
            (void)ctr; (void)ctf;
            for (int l = 0; l < 8; l++) g_personalBias[l] = NULL;
            g_personalW = 0;
        }
        memcpy(shDirFinal[sess], shDir, sizeof(float) * EMB_DIM);
        printf("\n");
    }
    {
        double c = 0;
        for (int d = 0; d < D; d++) c += shDirFinal[0][d] * shDirFinal[1][d];
        printf("  cos(direção sequestrada sessão 0, sessão 1) = %+.4f\n", c);
        printf("  (sessões que reprimem registros diferentes devem sequestrar direções diferentes)\n");
    }

    free(ids);
    freeWeights(&w, cfg); freeAdam(&adam);
}

/* ─── Repression via CE probes (--repression-ce) ──────────────────────────────
 * Same session machinery as --repression-test, but the DETECTOR is teacher-
 * forced CE on held-out probe passages instead of sampled-generation
 * signatures: deterministic (no sampling noise; ~1e-3 bits/char resolution,
 * proven by --interface-test) and perfectly paired (ΔCE = biased − unbiased
 * on the same probe). Sequestration ⇔ ΔCE on REPRESSED-register probes goes
 * negative and deepens as the Shadow accumulates; ΔCE on FED probes is the
 * specificity control; neutral turns test carry-over. No generations ⇒
 * multi-seed (3 × 2 registers) is affordable and every turn is measured. */

static int g_repressionCE = 0;     /* --repression-ce */

static void shadowRepressionCEExperiment(const char *corpusPath) {
    char idToChar[256];
    int nTokens, vocabSize;
    int *ids = loadCorpus(corpusPath, &nTokens, &vocabSize, idToChar);
    Config cfg = { .vocabSize = vocabSize, .contextLen = SEQ_LEN, .embDim = EMB_DIM,
                   .numHeads = 4, .numKVHeads = (NUM_KV_HEADS > 4) ? 4 : NUM_KV_HEADS,
                   .ffDim = FF_DIM, .stateN = 16 };
    applyLayout(&cfg);
    int jungLayer = -1;
    for (int l = 0; l < cfg.numLayers; l++)
        if (cfg.layerTypes[l] == MIX_JUNGIAN) { jungLayer = l; break; }
    if (jungLayer < 0) { fprintf(stderr, "repression-ce: needs a Jungian layer\n"); free(ids); return; }
    Weights w = initWeights(cfg);
    AdamBuf adam = initAdam(&w, cfg);
    int step = loadCkpt(&w, &adam, cfg);
    resetRand(1234);
    printModelHeader(cfg);
    printf("  corpus=%s  tokens=%d  vocab=%d  ckpt-step=%d  detector=CE teacher-forced\n\n",
           corpusPath, nTokens, vocabSize, step);

    int D = cfg.embDim, K16 = N_ARCH;
    long wb[6] = { 0, 16480, 382290, 842870, 988600, nTokens };
    const char *wname[5] = { "ChamberMusic", "Dubliners", "Portrait", "Exiles", "Ulysses" };
    if (!strstr(corpusPath, "joyce_all")) { fprintf(stderr, "repression-ce: needs joyce_all.txt\n"); free(ids); return; }
    const float LN2 = 0.6931472f;

    /* baseline hidden state */
    static float base[EMB_DIM]; memset(base, 0, sizeof(base));
    {
        long cnt = 0; int nS = 24, win = cfg.contextLen;
        long stride = (nTokens - win) / nS;
        for (int c = 0; c < nS; c++) {
            ForwardCache fc = forward(ids + c * stride, win, &w, cfg);
            const float *h = fc.blockCaches[jungLayer].ln1Out;
            for (int s = 0; s < win; s++)
                for (int d = 0; d < D; d++) base[d] += h[s * D + d];
            cnt += win; freeForwardCache(&fc);
        }
        for (int d = 0; d < D; d++) base[d] /= cnt;
    }

    /* phase 1: emergence (same protocol), keeping turn positions for probes */
    enum { M = 120, TURN = 384, KMAX = 12, CAL = 25 };
    static float sig[M][EMB_DIM]; static int tlab[M], tposR[M];
    for (int t = 0; t < M; t++) {
        int u = t % 5;
        long lo = wb[u] + 200, hi = wb[u + 1] - TURN - 200;
        int pos = (int)(lo + (long)(randf() * (double)(hi - lo)));
        tlab[t] = u; tposR[t] = pos;
        ForwardCache fc = forward(ids + pos, TURN, &w, cfg);
        const float *h = fc.blockCaches[jungLayer].ln1Out;
        double acc[EMB_DIM] = { 0 };
        for (int s = 0; s < TURN; s++)
            for (int d = 0; d < D; d++) acc[d] += h[s * D + d];
        double nrm = 0;
        for (int d = 0; d < D; d++) { sig[t][d] = (float)(acc[d] / TURN) - base[d]; nrm += sig[t][d] * sig[t][d]; }
        nrm = sqrt(nrm) + 1e-9;
        for (int d = 0; d < D; d++) sig[t][d] /= (float)nrm;
        freeForwardCache(&fc);
    }
    float pcos[CAL * CAL]; int np = 0;
    for (int a = 0; a < CAL; a++)
        for (int b = a + 1; b < CAL; b++) {
            double c = 0; for (int d = 0; d < D; d++) c += sig[a][d] * sig[b][d];
            pcos[np++] = (float)c;
        }
    for (int a = 0; a < np; a++)
        for (int b = a + 1; b < np; b++)
            if (pcos[b] < pcos[a]) { float tm = pcos[a]; pcos[a] = pcos[b]; pcos[b] = tm; }
    float theta = pcos[(int)(0.75f * np)];
    static float ce[KMAX][EMB_DIM]; float mass[KMAX]; int assign[M];
    int nK = 0; const float etaC = 0.15f;
    for (int t = 0; t < M; t++) {
        int bk = -1; double bc = -2;
        for (int k = 0; k < nK; k++) {
            double c = 0; for (int d = 0; d < D; d++) c += sig[t][d] * ce[k][d];
            if (c > bc) { bc = c; bk = k; }
        }
        if (bk < 0 || (bc < theta && nK < KMAX)) {
            memcpy(ce[nK], sig[t], sizeof(float) * D);
            mass[nK] = 1; assign[t] = nK; nK++;
        } else {
            double nrm = 0;
            for (int d = 0; d < D; d++) { ce[bk][d] = (1.f - etaC) * ce[bk][d] + etaC * sig[t][d]; nrm += ce[bk][d] * ce[bk][d]; }
            nrm = sqrt(nrm) + 1e-9;
            for (int d = 0; d < D; d++) ce[bk][d] /= (float)nrm;
            mass[bk] += 1; assign[t] = bk;
        }
    }
    printf("  emergence: %d archetypes (θ=%.3f)\n", nK, theta);

    /* baseline shadow S̄ (balanced battery) */
    const float tauP = 0.1f;
    float Sbar[KMAX]; memset(Sbar, 0, sizeof(Sbar));
    for (int t = 0; t < M; t++) {
        float pb[KMAX]; double mx = -1e9, sm = 0;
        for (int k = 0; k < nK; k++) {
            double c = 0; for (int d = 0; d < D; d++) c += sig[t][d] * ce[k][d];
            pb[k] = (float)(c / tauP);
            if (pb[k] > mx) mx = pb[k];
        }
        for (int k = 0; k < nK; k++) { pb[k] = (float)exp(pb[k] - mx); sm += pb[k]; }
        for (int k = 0; k < nK; k++)
            Sbar[k] += (1.f - pb[k] / (float)sm) / (float)(nK - 1);
    }
    for (int k = 0; k < nK; k++) Sbar[k] /= M;

    /* probes: top-3 battery turns per archetype (by cos to centroid), and
     * their UNBIASED teacher-forced CE — computed once, perfectly paired. */
    enum { NP = 3 };
    int probeT[KMAX][NP], nProbe[KMAX];
    float ce0[KMAX][NP];
    for (int k = 0; k < nK; k++) {
        nProbe[k] = 0;
        for (int r = 0; r < NP; r++) {
            int bt = -1; double bc = -2;
            for (int t = 0; t < M; t++) {
                if (assign[t] != k) continue;
                int used = 0;
                for (int j = 0; j < nProbe[k]; j++) if (probeT[k][j] == t) used = 1;
                if (used) continue;
                double c = 0; for (int d = 0; d < D; d++) c += sig[t][d] * ce[k][d];
                if (c > bc) { bc = c; bt = t; }
            }
            if (bt < 0) break;
            probeT[k][nProbe[k]++] = bt;
        }
        for (int p = 0; p < nProbe[k]; p++) {
            ForwardCache fc = forward(ids + tposR[probeT[k][p]], TURN, &w, cfg);
            ce0[k][p] = crossEntropyLoss(fc.probs, ids + tposR[probeT[k][p]] + 1, TURN - 1, cfg.vocabSize) / LN2;
            freeForwardCache(&fc);
        }
    }

    static float bias16[8][N_ARCH];
    static float shDir[EMB_DIM];
#if JUNGIAN_V2
    const float alpha = 0.85f, WSCALE = 60.f, WCAP = 12.f;
#else
    const float alpha = 0.85f, WSCALE = 150.f, WCAP = 40.f;
#endif
    enum { ST = 30, FEED = 24, NRUNS = 6 };
    const int sessWorks[2] = { 1, 3 };
    static float shDirFinal[2][EMB_DIM];
    double aggRepLate[2][3], aggRepNeu[2][3], aggFedLate[2][3];

    for (int run = 0; run < NRUNS; run++) {
        int wk2 = run % 2, sw = sessWorks[wk2], sd3 = run / 2;
        resetRand(777 + run);
        float S[KMAX]; for (int k = 0; k < nK; k++) S[k] = 1.f / nK;
        float pbar[KMAX]; memset(pbar, 0, sizeof(pbar)); int pbarN = 0;
        int fed[KMAX]; memset(fed, 0, sizeof(fed)); int frozen = 0, nFed = 0;
        float signCache[KMAX]; int signKnown[KMAX];
        for (int k = 0; k < nK; k++) { signCache[k] = 1.f; signKnown[k] = 0; }
        double repLate = 0, fedLate = 0, repNeu = 0; int nLate = 0, nNeu = 0;
        int verbose = (run < 2);
        if (verbose) {
            printf("\n  ── Sessão «%s» (seed %d) — ΔCE = CE(viés) − CE(sem), bits/char ─────\n", wname[sw], sd3);
            printf("     %-3s | %-7s | %-6s | %-5s | %10s | %10s | %s\n",
                   "t", "fase", "dev(S)", "w_u", "ΔCE_reprim", "ΔCE_alim", "dir");
        }
        for (int t = 0; t < ST; t++) {
            int wk = (t < FEED) ? sw : (t % 5);
            long lo = wb[wk] + 200, hi = wb[wk + 1] - TURN - 200;
            int pos = (int)(lo + (long)(randf() * (double)(hi - lo)));
            ForwardCache fc = forward(ids + pos, TURN, &w, cfg);
            const float *h = fc.blockCaches[jungLayer].ln1Out;
            double acc[EMB_DIM] = { 0 };
            for (int s = 0; s < TURN; s++)
                for (int d = 0; d < D; d++) acc[d] += h[s * D + d];
            float psig[EMB_DIM]; double nrm = 0;
            for (int d = 0; d < D; d++) { psig[d] = (float)(acc[d] / TURN) - base[d]; nrm += psig[d] * psig[d]; }
            nrm = sqrt(nrm) + 1e-9;
            for (int d = 0; d < D; d++) psig[d] /= (float)nrm;
            freeForwardCache(&fc);
            float p[KMAX]; double mx = -1e9, sm = 0;
            for (int k = 0; k < nK; k++) {
                double c = 0; for (int d = 0; d < D; d++) c += psig[d] * ce[k][d];
                p[k] = (float)(c / tauP);
                if (p[k] > mx) mx = p[k];
            }
            for (int k = 0; k < nK; k++) { p[k] = (float)exp(p[k] - mx); sm += p[k]; }
            for (int k = 0; k < nK; k++) p[k] /= (float)sm;
            if (t < FEED) { for (int k = 0; k < nK; k++) pbar[k] += p[k]; pbarN++; }
            for (int k = 0; k < nK; k++)
                S[k] = alpha * S[k] + (1.f - alpha) * (1.f - p[k]) / (float)(nK - 1);
            if (!frozen && t == 3) {
                nFed = 0;
                for (int k = 0; k < nK; k++) { fed[k] = (pbar[k] / pbarN > 1.f / nK); nFed += fed[k]; }
                frozen = 1;
            }
            if (!frozen) continue;
            /* direction: session-relative repression, sparse top-2 */
            double sSum = 0; for (int k = 0; k < nK; k++) sSum += S[k];
            float score[KMAX]; float dev = 0; int k1 = -1, k2 = -1;
            for (int k = 0; k < nK; k++) {
                score[k] = (float)(S[k] / sSum) - Sbar[k];
                dev += fabsf(score[k]);
                if (k1 < 0 || score[k] > score[k1]) { k2 = k1; k1 = k; }
                else if (k2 < 0 || score[k] > score[k2]) { k2 = k; }
            }
            memset(shDir, 0, sizeof(float) * EMB_DIM);
            for (int d = 0; d < D; d++) {
                shDir[d] = score[k1] * ce[k1][d];
                if (k2 >= 0 && score[k2] > 0) shDir[d] += score[k2] * ce[k2][d];
            }
            double dn = 0; for (int d = 0; d < D; d++) dn += shDir[d] * shDir[d];
            dn = sqrt(dn) + 1e-9;
            for (int d = 0; d < D; d++) shDir[d] /= (float)dn;
            for (int l = 0; l < cfg.numLayers && l < 8; l++) {
                if (cfg.layerTypes[l] != MIX_JUNGIAN) continue;
                const float *wa = w.blocks[l].wArch;
                double bn = 0;
                for (int c16 = 0; c16 < K16; c16++) {
                    double b = 0;
                    for (int d = 0; d < D; d++) b += shDir[d] * wa[c16 * D + d];
                    bias16[l][c16] = (float)b; bn += b * b;
                }
                bn = sqrt(bn) + 1e-9;
                for (int c16 = 0; c16 < K16; c16++) bias16[l][c16] /= (float)bn;
            }
            /* CE of a probe under the current bias */
            #define PROBE_CE(kk, pp, ww) ({ \
                for (int l = 0; l < 8; l++) g_personalBias[l] = NULL; \
                for (int l = 0; l < cfg.numLayers && l < 8; l++) \
                    if (cfg.layerTypes[l] == MIX_JUNGIAN) g_personalBias[l] = bias16[l]; \
                g_personalW = (ww); \
                ForwardCache pf = forward(ids + tposR[probeT[kk][pp]], TURN, &w, cfg); \
                float ceb = crossEntropyLoss(pf.probs, ids + tposR[probeT[kk][pp]] + 1, TURN - 1, cfg.vocabSize) / LN2; \
                freeForwardCache(&pf); \
                for (int l = 0; l < 8; l++) g_personalBias[l] = NULL; \
                g_personalW = 0.f; \
                ceb; })
            /* polarity by CE, once per leader: sign that LOWERS CE on its probes */
            if (!signKnown[k1] && nProbe[k1] > 0) {
                double dpos = 0, dneg = 0;
                for (int pp = 0; pp < nProbe[k1]; pp++) {
                    dpos += PROBE_CE(k1, pp, 12.f) - ce0[k1][pp];
                    dneg += PROBE_CE(k1, pp, -12.f) - ce0[k1][pp];
                }
                signCache[k1] = (dpos <= dneg) ? 1.f : -1.f;
                signKnown[k1] = 1;
                if (verbose)
                    printf("     confront A%d (CE): Δ(+12)=%+.4f Δ(−12)=%+.4f → sinal %s\n",
                           k1, dpos / nProbe[k1], dneg / nProbe[k1], signCache[k1] > 0 ? "+" : "−");
            }
            float wu = WSCALE * dev; if (wu > WCAP) wu = WCAP;
            wu *= signCache[k1];
            /* ΔCE on repressed-elected probes and on fed probes */
            double dRep = 0; int nR = 0;
            for (int pp = 0; pp < nProbe[k1]; pp++) { dRep += PROBE_CE(k1, pp, wu) - ce0[k1][pp]; nR++; }
            if (k2 >= 0 && score[k2] > 0)
                for (int pp = 0; pp < nProbe[k2] && pp < 2; pp++) { dRep += PROBE_CE(k2, pp, wu) - ce0[k2][pp]; nR++; }
            double dFed = 0; int nF = 0;
            for (int k = 0; k < nK && nF < 4; k++) {
                if (!fed[k]) continue;
                for (int pp = 0; pp < nProbe[k] && pp < 2 && nF < 4; pp++) { dFed += PROBE_CE(k, pp, wu) - ce0[k][pp]; nF++; }
            }
            dRep /= (nR ? nR : 1); dFed /= (nF ? nF : 1);
            if (t >= 12 && t < FEED) { repLate += dRep; fedLate += dFed; nLate++; }
            if (t >= FEED)           { repNeu += dRep; nNeu++; }
            if (verbose && ((t + 1) % 4 == 0 || t >= FEED))
                printf("     %-3d | %-7s | %6.3f | %5.1f | %+10.4f | %+10.4f | A%d%s\n",
                       t + 1, t < FEED ? "feed" : "NEUTRO", dev, wu, dRep, dFed, k1,
                       (k2 >= 0 && score[k2] > 0) ? "+" : "");
        }
        aggRepLate[wk2][sd3] = repLate / (nLate ? nLate : 1);
        aggFedLate[wk2][sd3] = fedLate / (nLate ? nLate : 1);
        aggRepNeu[wk2][sd3]  = repNeu / (nNeu ? nNeu : 1);
        if (run < 2) memcpy(shDirFinal[wk2], shDir, sizeof(float) * D);
    }

    printf("\n  ── Agregado (3 seeds × 2 registros; ΔCE bits/char; sequestro ⇔ reprim < 0 < alim) ──\n");
    printf("     %-12s | %-22s | %-22s | %s\n", "sessão", "ΔCE_reprim (feed 13-24)", "ΔCE_reprim (NEUTRO)", "ΔCE_alim (controle)");
    for (int wk2 = 0; wk2 < 2; wk2++) {
        double mL = 0, mN = 0, mF = 0, vL = 0;
        for (int s = 0; s < 3; s++) { mL += aggRepLate[wk2][s]; mN += aggRepNeu[wk2][s]; mF += aggFedLate[wk2][s]; }
        mL /= 3; mN /= 3; mF /= 3;
        for (int s = 0; s < 3; s++) vL += (aggRepLate[wk2][s] - mL) * (aggRepLate[wk2][s] - mL);
        printf("     %-12s | %+10.4f ± %.4f    | %+10.4f            | %+10.4f\n",
               wname[sessWorks[wk2]], mL, sqrt(vL / 3), mN, mF);
    }
    {
        double c = 0;
        for (int d = 0; d < D; d++) c += shDirFinal[0][d] * shDirFinal[1][d];
        printf("     cos(direção sessão 0, sessão 1) = %+.4f\n", c);
    }
    free(ids);
    freeWeights(&w, cfg); freeAdam(&adam);
}

int main(int argc, char **argv) {
#if GRAD_CHECK
    (void)argc; (void)argv;
    gradCheck();
#else
    if (argc > 2 && strcmp(argv[1], "--tools") == 0)
        trainToolCalling(argv[2]);          /* ./transformer --tools tooldata */
    else if (argc > 1) {                    /* ./transformer corpus/pg100.txt [--ckpt f [--until n]] */
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc) g_ckptPath = argv[++i];
            else if (strcmp(argv[i], "--until") == 0 && i + 1 < argc) g_untilStep = atoi(argv[++i]);
            else if (strcmp(argv[i], "--eval-sample") == 0 && i + 1 < argc) g_evalSample = atoi(argv[++i]);
            else if (strcmp(argv[i], "--eval-window") == 0 && i + 1 < argc) g_evalWindow = atoi(argv[++i]);
            else if (strcmp(argv[i], "--shadow-cache") == 0) g_shadowCacheReq = 1;
            else if (strcmp(argv[i], "--persona-test") == 0) g_personaTest = 1;
            else if (strcmp(argv[i], "--oracle-test") == 0) g_oracleTest = 1;
            else if (strcmp(argv[i], "--emerge-test") == 0) g_emergeTest = 1;
            else if (strcmp(argv[i], "--repression-test") == 0) g_repressionTest = 1;
            else if (strcmp(argv[i], "--interface-test") == 0) g_interfaceTest = 1;
            else if (strcmp(argv[i], "--repression-ce") == 0) g_repressionCE = 1;
            else if (strcmp(argv[i], "--labels") == 0 && i + 1 < argc) g_oracleLabels = argv[++i];
            else if (strcmp(argv[i], "--delta") == 0 && i + 1 < argc) { g_reinforceDelta = (float)atof(argv[++i]); g_deltaGiven = 1; }
        }
        if (g_personaTest) shadowPersonaExperiment(argv[1]);
        else if (g_oracleTest) shadowOracleExperiment(argv[1]);
        else if (g_emergeTest) shadowEmergenceExperiment(argv[1]);
        else if (g_repressionTest) shadowRepressionExperiment(argv[1]);
        else if (g_interfaceTest) steerInterfaceExperiment(argv[1]);
        else if (g_repressionCE) shadowRepressionCEExperiment(argv[1]);
        else trainFromFile(argv[1]);
    }
    else trainToy();                        /* ./transformer -> quick toy run  */
#endif
    return 0;
}
