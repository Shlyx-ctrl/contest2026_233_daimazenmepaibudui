/*
 * sound_classifier.c - 纯声音分类（不含业务逻辑）
 */

#include "sound_classifier.h"
#include "sound_model_weights.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static bool initialized = false;

/* ═══════════════════════════════════════════════════════════════
 * MFCC 特征提取
 * ═══════════════════════════════════════════════════════════════ */

static inline float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static int extract_mfcc(const int16_t *audio, int len,
                        float *features, int *out_frames) {
    if (len < SOUND_N_FFT) return -1;

    int n_frames = (len - SOUND_N_FFT) / SOUND_HOP_LENGTH + 1;
    if (n_frames <= 0) return -1;
    if (n_frames > 94) n_frames = 94;

    *out_frames = n_frames;

    /* Mel 滤波器组 */
    float mel_low = hz_to_mel(0);
    float mel_high = hz_to_mel(SOUND_SAMPLE_RATE / 2.0f);
    float *mel_fb = (float *)calloc(SOUND_N_MFCC * (SOUND_N_FFT / 2 + 1), sizeof(float));
    if (!mel_fb) return -1;

    for (int i = 0; i < SOUND_N_MFCC; i++) {
        float fc = mel_to_hz(mel_low + (mel_high - mel_low) * (i + 1) / (SOUND_N_MFCC + 1));
        int fc_bin = (int)(fc * SOUND_N_FFT / SOUND_SAMPLE_RATE + 0.5f);
        int start = (i > 0) ? (int)(mel_to_hz(mel_low + (mel_high - mel_low) * i / (SOUND_N_MFCC + 1)) * SOUND_N_FFT / SOUND_SAMPLE_RATE) : 0;
        int end = (i < SOUND_N_MFCC - 1) ? (int)(mel_to_hz(mel_low + (mel_high - mel_low) * (i + 2) / (SOUND_N_MFCC + 1)) * SOUND_N_FFT / SOUND_SAMPLE_RATE + 0.5f) : SOUND_N_FFT / 2;

        for (int j = start; j <= end && j <= SOUND_N_FFT / 2; j++) {
            if (j < fc_bin)
                mel_fb[i * (SOUND_N_FFT / 2 + 1) + j] = (float)(j - start) / (fc_bin - start + 1);
            else
                mel_fb[i * (SOUND_N_FFT / 2 + 1) + j] = (float)(end - j) / (end - fc_bin + 1);
        }
    }

    /* MFCC */
    float *mfcc_all = (float *)malloc(SOUND_N_MFCC * n_frames * sizeof(float));
    if (!mfcc_all) { free(mel_fb); return -1; }

    for (int t = 0; t < n_frames; t++) {
        int offset = t * SOUND_HOP_LENGTH;
        float power[SOUND_N_FFT / 2 + 1] = {0};

        for (int k = 0; k < SOUND_N_FFT / 2; k++) {
            float re = 0, im = 0;
            for (int n = 0; n < SOUND_N_FFT; n++) {
                float s = audio[offset + n] / 32768.0f;
                float a = 2.0f * M_PI * k * n / SOUND_N_FFT;
                re += s * cosf(a);
                im -= s * sinf(a);
            }
            power[k] = (re * re + im * im) / SOUND_N_FFT;
        }

        for (int m = 0; m < SOUND_N_MFCC; m++) {
            float sum = 0;
            for (int k = 0; k <= SOUND_N_FFT / 2; k++)
                sum += power[k] * mel_fb[m * (SOUND_N_FFT / 2 + 1) + k];
            float log_mel = logf(sum + 1e-10f);
            float val = 0;
            for (int n = 0; n < SOUND_N_MFCC; n++)
                val += log_mel * cosf(M_PI * m * (2 * n + 1) / (2.0f * SOUND_N_MFCC));
            mfcc_all[t * SOUND_N_MFCC + m] = val;
        }
    }
    free(mel_fb);

    /* MFCC + delta + delta2 */
    for (int t = 0; t < n_frames; t++) {
        for (int m = 0; m < SOUND_N_MFCC; m++) {
            float cur = mfcc_all[t * SOUND_N_MFCC + m];
            features[0 * SOUND_N_MFCC * n_frames + t * SOUND_N_MFCC + m] = cur;
            float prev = (t > 0) ? mfcc_all[(t-1) * SOUND_N_MFCC + m] : cur;
            float next = (t < n_frames-1) ? mfcc_all[(t+1) * SOUND_N_MFCC + m] : cur;
            features[1 * SOUND_N_MFCC * n_frames + t * SOUND_N_MFCC + m] = (next - prev) / 2.0f;
            float prev2 = (t > 1) ? mfcc_all[(t-2) * SOUND_N_MFCC + m] : prev;
            float next2 = (t < n_frames-2) ? mfcc_all[(t+2) * SOUND_N_MFCC + m] : next;
            features[2 * SOUND_N_MFCC * n_frames + t * SOUND_N_MFCC + m] = prev2 - 2.0f * cur + next2;
        }
    }
    free(mfcc_all);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * CNN 推理
 * ═══════════════════════════════════════════════════════════════ */

static inline float relu(float x) { return x > 0 ? x : 0; }

static void conv2d_fwd(const float *in, const float *w, const float *b,
                       float *out, int ic, int oc, int h, int ws) {
    for (int o = 0; o < oc; o++)
        for (int oh = 0; oh < h; oh++)
            for (int ow = 0; ow < ws; ow++) {
                float sum = b[o];
                for (int c = 0; c < ic; c++)
                    for (int kh = 0; kh < 3; kh++)
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh + kh - 1, iw = ow + kw - 1;
                            if (ih >= 0 && ih < h && iw >= 0 && iw < ws)
                                sum += in[c*h*ws + ih*ws + iw] * w[o*ic*9 + c*9 + kh*3 + kw];
                        }
                out[o*h*ws + oh*ws + ow] = relu(sum);
            }
}

static void maxpool_fwd(const float *in, float *out, int c, int h, int w) {
    for (int ch = 0; ch < c; ch++)
        for (int oh = 0; oh < h/2; oh++)
            for (int ow = 0; ow < w/2; ow++) {
                float mx = -1e9f;
                for (int kh = 0; kh < 2; kh++)
                    for (int kw = 0; kw < 2; kw++) {
                        float v = in[ch*h*w + (oh*2+kh)*w + (ow*2+kw)];
                        if (v > mx) mx = v;
                    }
                out[ch*(h/2)*(w/2) + oh*(w/2) + ow] = mx;
            }
}

static void avgpool_fwd(const float *in, float *out, int c, int h, int w) {
    for (int ch = 0; ch < c; ch++) {
        float sum = 0;
        for (int i = 0; i < h*w; i++) sum += in[ch*h*w + i];
        out[ch] = sum / (h*w);
    }
}

static void fc_fwd(const float *in, const float *w, const float *b,
                   float *out, int inf, int outf) {
    for (int o = 0; o < outf; o++) {
        float sum = b[o];
        for (int i = 0; i < inf; i++) sum += in[i] * w[o*inf + i];
        out[o] = sum;
    }
}

static void softmax_fwd(const float *in, float *out, int n) {
    float mx = in[0];
    for (int i = 1; i < n; i++) if (in[i] > mx) mx = in[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { out[i] = expf(in[i] - mx); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

static int forward_cnn(const float *features, int n_frames, float *probs) {
    int H = 40, W = (n_frames > 94) ? 94 : n_frames;

    float *b1 = malloc(32 * H * W * sizeof(float));
    float *b2 = malloc(32 * (H/2) * (W/2) * sizeof(float));
    float *b3 = malloc(64 * (H/2) * (W/2) * sizeof(float));
    float *b4 = malloc(64 * (H/4) * (W/4) * sizeof(float));
    float *b5 = malloc(128 * (H/4) * (W/4) * sizeof(float));
    float *pool = malloc(128 * sizeof(float));
    float *fc1 = malloc(256 * sizeof(float));
    if (!b1||!b2||!b3||!b4||!b5||!pool||!fc1) {
        free(b1);free(b2);free(b3);free(b4);free(b5);free(pool);free(fc1);
        return -1;
    }

    conv2d_fwd(features, sound_model_conv1_weight, sound_model_conv1_bias, b1, 3, 32, H, W);
    maxpool_fwd(b1, b2, 32, H, W);
    conv2d_fwd(b2, sound_model_conv2_weight, sound_model_conv2_bias, b3, 32, 64, H/2, W/2);
    maxpool_fwd(b3, b4, 64, H/2, W/2);
    conv2d_fwd(b4, sound_model_conv3_weight, sound_model_conv3_bias, b5, 64, 128, H/4, W/4);
    avgpool_fwd(b5, pool, 128, H/4, W/4);
    fc_fwd(pool, sound_model_fc1_weight, sound_model_fc1_bias, fc1, 128, 256);
    float logits[8];
    fc_fwd(fc1, sound_model_fc2_weight, sound_model_fc2_bias, logits, 256, 8);
    softmax_fwd(logits, probs, 8);

    free(b1);free(b2);free(b3);free(b4);free(b5);free(pool);free(fc1);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════
 * 公共 API
 * ═══════════════════════════════════════════════════════════════ */

int sound_classifier_init(void) {
    initialized = true;
    return 0;
}

int sound_classifier_run(const int16_t *audio, int len, sound_result_t *result) {
    if (!initialized || !audio || !result) return -1;

    float features[SOUND_N_CHANNELS * SOUND_N_MFCC * 94];
    int n_frames = 0;
    if (extract_mfcc(audio, len, features, &n_frames) != 0) return -1;

    if (forward_cnn(features, n_frames, result->probs) != 0) return -1;

    int best = 0;
    for (int i = 1; i < SOUND_MODEL_NUM_CLASSES; i++) {
        if (result->probs[i] > result->probs[best]) best = i;
    }
    result->label = best;
    result->confidence = result->probs[best];
    return 0;
}

void sound_classifier_deinit(void) {
    initialized = false;
}
