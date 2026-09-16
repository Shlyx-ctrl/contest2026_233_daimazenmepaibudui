/*
 * sound_classifier.c - 智爱陪伴 音频分类器
 *
 * 功能：MFCC + CNN 推理 + 多模态融合
 */

#include "sound_classifier.h"
#include "sound_model_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static bool initialized = false;
static sound_fusion_t *s_fusion = NULL;

/*═══════════════════════════════════════════════════════════════════
 * MFCC 特征提取
 *═══════════════════════════════════════════════════════════════════*/

static inline float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/**
 * 计算 MFCC + delta + delta2
 * 输出: [SOUND_N_CHANNELS][SOUND_N_MFCC][n_frames]
 */
static int extract_mfcc(const int16_t *audio, int len,
                        float *features, int *out_frames) {
    if (len < SOUND_N_FFT) return -1;

    int n_frames = (len - SOUND_N_FFT) / SOUND_HOP_LENGTH + 1;
    if (n_frames <= 0) return -1;
    if (n_frames > 600) n_frames = 600;

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
            if (j < fc_bin) {
                mel_fb[i * (SOUND_N_FFT / 2 + 1) + j] = (float)(j - start) / (fc_bin - start + 1);
            } else {
                mel_fb[i * (SOUND_N_FFT / 2 + 1) + j] = (float)(end - j) / (end - fc_bin + 1);
            }
        }
    }

    /* MFCC 提取 */
    float *mfcc_all = (float *)malloc(SOUND_N_MFCC * n_frames * sizeof(float));
    if (!mfcc_all) { free(mel_fb); return -1; }

    for (int t = 0; t < n_frames; t++) {
        int offset = t * SOUND_HOP_LENGTH;

        /* 简化的 FFT: 直接计算功率谱 */
        float power[SOUND_N_FFT / 2 + 1] = {0};
        for (int k = 0; k < SOUND_N_FFT / 2; k++) {
            float re = 0, im = 0;
            for (int n = 0; n < SOUND_N_FFT; n++) {
                float sample = audio[offset + n] / 32768.0f;
                float angle = 2.0f * M_PI * k * n / SOUND_N_FFT;
                re += sample * cosf(angle);
                im -= sample * sinf(angle);
            }
            power[k] = (re * re + im * im) / SOUND_N_FFT;
        }

        /* Mel 滤波 */
        for (int m = 0; m < SOUND_N_MFCC; m++) {
            float sum = 0;
            for (int k = 0; k <= SOUND_N_FFT / 2; k++) {
                sum += power[k] * mel_fb[m * (SOUND_N_FFT / 2 + 1) + k];
            }
            float log_mel = logf(sum + 1e-10f);

            /* DCT */
            float val = 0;
            for (int n = 0; n < SOUND_N_MFCC; n++) {
                val += log_mel * cosf(M_PI * m * (2 * n + 1) / (2.0f * SOUND_N_MFCC));
            }
            mfcc_all[t * SOUND_N_MFCC + m] = val;
        }
    }
    free(mel_fb);

    /* 计算 delta 和 delta2 */
    for (int t = 0; t < n_frames; t++) {
        for (int m = 0; m < SOUND_N_MFCC; m++) {
            float cur = mfcc_all[t * SOUND_N_MFCC + m];

            /* MFCC (channel 0) */
            features[0 * SOUND_N_MFCC * n_frames + t * SOUND_N_MFCC + m] = cur;

            /* Delta (channel 1) */
            float prev = (t > 0) ? mfcc_all[(t-1) * SOUND_N_MFCC + m] : cur;
            float next = (t < n_frames-1) ? mfcc_all[(t+1) * SOUND_N_MFCC + m] : cur;
            features[1 * SOUND_N_MFCC * n_frames + t * SOUND_N_MFCC + m] = (next - prev) / 2.0f;

            /* Delta2 (channel 2) */
            float prev2 = (t > 1) ? mfcc_all[(t-2) * SOUND_N_MFCC + m] : prev;
            float next2 = (t < n_frames-2) ? mfcc_all[(t+2) * SOUND_N_MFCC + m] : next;
            features[2 * SOUND_N_MFCC * n_frames + t * SOUND_N_MFCC + m] = prev2 - 2.0f * cur + next2;
        }
    }

    free(mfcc_all);
    return 0;
}

/*═══════════════════════════════════════════════════════════════════
 * CNN 推理
 *═══════════════════════════════════════════════════════════════════*/

static inline float relu(float x) { return x > 0 ? x : 0; }

static void conv2d_forward(const float *in, const float *w, const float *b,
                           float *out, int ic, int oc, int h, int w_size) {
    for (int o = 0; o < oc; o++) {
        for (int oh = 0; oh < h; oh++) {
            for (int ow = 0; ow < w_size; ow++) {
                float sum = b[o];
                for (int c = 0; c < ic; c++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh + kh - 1;
                            int iw = ow + kw - 1;
                            if (ih >= 0 && ih < h && iw >= 0 && iw < w_size) {
                                sum += in[c * h * w_size + ih * w_size + iw] *
                                       w[o * ic * 9 + c * 9 + kh * 3 + kw];
                            }
                        }
                    }
                }
                out[o * h * w_size + oh * w_size + ow] = relu(sum);
            }
        }
    }
}

static void maxpool2d_forward(const float *in, float *out, int c, int h, int w) {
    for (int ch = 0; ch < c; ch++) {
        for (int oh = 0; oh < h/2; oh++) {
            for (int ow = 0; ow < w/2; ow++) {
                float mx = -1e9f;
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        float v = in[ch * h * w + (oh*2+kh) * w + (ow*2+kw)];
                        if (v > mx) mx = v;
                    }
                }
                out[ch * (h/2) * (w/2) + oh * (w/2) + ow] = mx;
            }
        }
    }
}

static void avgpool_forward(const float *in, float *out, int c, int h, int w) {
    for (int ch = 0; ch < c; ch++) {
        float sum = 0;
        for (int i = 0; i < h * w; i++) sum += in[ch * h * w + i];
        out[ch] = sum / (h * w);
    }
}

static void fc_forward(const float *in, const float *w, const float *b,
                       float *out, int in_f, int out_f) {
    for (int o = 0; o < out_f; o++) {
        float sum = b[o];
        for (int i = 0; i < in_f; i++) sum += in[i] * w[o * in_f + i];
        out[o] = sum;
    }
}

static void softmax_forward(const float *in, float *out, int n) {
    float mx = in[0];
    for (int i = 1; i < n; i++) if (in[i] > mx) mx = in[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { out[i] = expf(in[i] - mx); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

/**
 * CNN 前向推理
 * 输入: features [SOUND_N_CHANNELS][SOUND_N_MFCC][T]
 * 输出: probs [SOUND_MODEL_NUM_CLASSES]
 */
static int forward_cnn(const float *features, int n_frames, float *probs) {
    int H = SOUND_MODEL_INPUT_H;  /* 40 */
    int W = n_frames;
    if (W > 94) W = 94;

    float *b1 = (float *)malloc(SOUND_MODEL_CONV1_OUT * H * W * sizeof(float));
    float *b2 = (float *)malloc(SOUND_MODEL_CONV1_OUT * (H/2) * (W/2) * sizeof(float));
    float *b3 = (float *)malloc(SOUND_MODEL_CONV2_OUT * (H/2) * (W/2) * sizeof(float));
    float *b4 = (float *)malloc(SOUND_MODEL_CONV2_OUT * (H/4) * (W/4) * sizeof(float));
    float *b5 = (float *)malloc(SOUND_MODEL_CONV3_OUT * (H/4) * (W/4) * sizeof(float));
    float *pool = (float *)malloc(SOUND_MODEL_CONV3_OUT * sizeof(float));
    float *fc1_out = (float *)malloc(SOUND_MODEL_FC1_OUT * sizeof(float));

    if (!b1 || !b2 || !b3 || !b4 || !b5 || !pool || !fc1_out) {
        free(b1); free(b2); free(b3); free(b4); free(b5); free(pool); free(fc1_out);
        return -1;
    }

    conv2d_forward(features, sound_model_conv1_weight, sound_model_conv1_bias,
                   b1, SOUND_MODEL_INPUT_C, SOUND_MODEL_CONV1_OUT, H, W);
    maxpool2d_forward(b1, b2, SOUND_MODEL_CONV1_OUT, H, W);

    conv2d_forward(b2, sound_model_conv2_weight, sound_model_conv2_bias,
                   b3, SOUND_MODEL_CONV1_OUT, SOUND_MODEL_CONV2_OUT, H/2, W/2);
    maxpool2d_forward(b3, b4, SOUND_MODEL_CONV2_OUT, H/2, W/2);

    conv2d_forward(b4, sound_model_conv3_weight, sound_model_conv3_bias,
                   b5, SOUND_MODEL_CONV2_OUT, SOUND_MODEL_CONV3_OUT, H/4, W/4);
    avgpool_forward(b5, pool, SOUND_MODEL_CONV3_OUT, H/4, W/4);

    fc_forward(pool, sound_model_fc1_weight, sound_model_fc1_bias,
               fc1_out, SOUND_MODEL_CONV3_OUT, SOUND_MODEL_FC1_OUT);

    float logits[SOUND_MODEL_NUM_CLASSES];
    fc_forward(fc1_out, sound_model_fc2_weight, sound_model_fc2_bias,
               logits, SOUND_MODEL_FC1_OUT, SOUND_MODEL_NUM_CLASSES);
    softmax_forward(logits, probs, SOUND_MODEL_NUM_CLASSES);

    free(b1); free(b2); free(b3); free(b4); free(b5); free(pool); free(fc1_out);
    return 0;
}

/*═══════════════════════════════════════════════════════════════════
 * 多模态融合
 *═══════════════════════════════════════════════════════════════════*/

static float compute_rms(const int16_t *audio, int len) {
    if (!audio || len <= 0) return 0;
    int64_t sum = 0;
    for (int i = 0; i < len; i++) sum += (int32_t)audio[i] * audio[i];
    return sqrtf((float)sum / len);
}

static exec_level_t get_exec_level(float confidence, int label) {
    float th = (label == SOUND_LABEL_FALL) ? CONF_THRESHOLD_FALL : CONF_THRESHOLD_LOG;
    if (confidence < th) return EXEC_LEVEL_NONE;
    if (confidence < CONF_THRESHOLD_HIGH) return EXEC_LEVEL_NORMAL;
    if (confidence < CONF_THRESHOLD_URGENT) return EXEC_LEVEL_NORMAL;
    return EXEC_LEVEL_URGENT;
}

float sound_get_threshold(int label) {
    return (label == SOUND_LABEL_FALL) ? CONF_THRESHOLD_FALL : CONF_THRESHOLD_LOG;
}

static int time_window_vote(sound_fusion_t *ctx, int new_label) {
    ctx->history[ctx->history_idx] = new_label;
    ctx->history_idx = (ctx->history_idx + 1) % SOUND_HISTORY_SIZE;
    if (ctx->history_count < SOUND_HISTORY_SIZE) ctx->history_count++;
    if (ctx->history_count < 3) return -1;

    int counts[SOUND_LABEL_COUNT] = {0};
    for (int i = 0; i < ctx->history_count; i++) {
        if (ctx->history[i] >= 0 && ctx->history[i] < SOUND_LABEL_COUNT)
            counts[ctx->history[i]]++;
    }

    int best = -1, best_c = 0;
    for (int i = 0; i < SOUND_LABEL_COUNT; i++) {
        if (counts[i] > best_c) { best_c = counts[i]; best = i; }
    }
    return (best_c >= (ctx->history_count * 2) / 3) ? best : -1;
}

static void fall_detection_fsm(sound_fusion_t *ctx, int label, float conf,
                               int64_t now_ms, sound_result_t *result) {
    bool is_fall = (label == SOUND_LABEL_FALL && conf > CONF_THRESHOLD_FALL);

    switch (ctx->fall_state) {
    case FALL_STATE_IDLE:
        if (is_fall) {
            ctx->fall_state = FALL_STATE_SUSPECTED;
            ctx->fall_time_ms = now_ms;
            ctx->fall_votes = 1;
            if (ctx->tts_cb) ctx->tts_cb("您还好吗？请回答");
        }
        break;
    case FALL_STATE_SUSPECTED:
        if (is_fall) ctx->fall_votes++;
        if (now_ms - ctx->fall_time_ms > SOUND_FALL_TIMEOUT_MS) {
            if (ctx->fall_votes >= 2) {
                result->label = SOUND_LABEL_FALL;
                result->confidence = conf;
                result->exec_level = EXEC_LEVEL_URGENT;
                result->is_fused = true;
                ctx->fall_alarm_count++;
                if (ctx->alarm_cb) ctx->alarm_cb(1, "检测到跌倒！");
            }
            ctx->fall_state = FALL_STATE_IDLE;
        }
        break;
    case FALL_STATE_CONFIRMING:
        ctx->fall_state = FALL_STATE_IDLE;
        break;
    }
}

/*═══════════════════════════════════════════════════════════════════
 * 公共 API
 *═══════════════════════════════════════════════════════════════════*/

int sound_classifier_init(sound_fusion_t *fusion,
                          tts_callback_t tts_cb,
                          alarm_callback_t alarm_cb,
                          cloud_callback_t cloud_cb) {
    if (!fusion) return -1;
    memset(fusion, 0, sizeof(sound_fusion_t));
    fusion->tts_cb = tts_cb;
    fusion->alarm_cb = alarm_cb;
    fusion->cloud_cb = cloud_cb;
    for (int i = 0; i < SOUND_HISTORY_SIZE; i++) fusion->history[i] = -1;
    s_fusion = fusion;
    initialized = true;
    return 0;
}

int sound_classify_fused(const int16_t *audio, int64_t now_ms, sound_result_t *result) {
    if (!s_fusion || !audio || !result || !initialized) return -1;

    float features[SOUND_N_CHANNELS * SOUND_N_MFCC * 600];
    int n_frames = 0;
    if (extract_mfcc(audio, SOUND_BUFFER_SIZE, features, &n_frames) != 0) return -1;

    float probs[SOUND_MODEL_NUM_CLASSES];
    if (forward_cnn(features, n_frames, probs) != 0) return -1;

    int best = 0;
    for (int i = 1; i < SOUND_MODEL_NUM_CLASSES; i++) {
        if (probs[i] > probs[best]) best = i;
    }

    result->label = best;
    result->confidence = probs[best];
    result->exec_level = get_exec_level(probs[best], best);
    result->is_fused = false;
    result->vote_count = 1;
    s_fusion->total_detections++;

    int voted = time_window_vote(s_fusion, best);
    if (voted >= 0 && voted != best) {
        result->label = voted;
        result->is_fused = true;
        result->vote_count = 3;
    }

    if (result->label == SOUND_LABEL_FALL || best == SOUND_LABEL_FALL)
        fall_detection_fsm(s_fusion, best, probs[best], now_ms, result);

    if (result->exec_level == EXEC_LEVEL_URGENT) s_fusion->urgent_count++;
    return 0;
}

void sound_get_stats(int *total, int *urgent, int *fall_alarms) {
    if (!s_fusion) return;
    if (total) *total = s_fusion->total_detections;
    if (urgent) *urgent = s_fusion->urgent_count;
    if (fall_alarms) *fall_alarms = s_fusion->fall_alarm_count;
}

void sound_classifier_reset(void) {
    if (!s_fusion) return;
    s_fusion->history_count = 0;
    s_fusion->history_idx = 0;
    s_fusion->fall_state = FALL_STATE_IDLE;
    for (int i = 0; i < SOUND_HISTORY_SIZE; i++) s_fusion->history[i] = -1;
}

void sound_classifier_deinit(void) {
    initialized = false;
    s_fusion = NULL;
}
