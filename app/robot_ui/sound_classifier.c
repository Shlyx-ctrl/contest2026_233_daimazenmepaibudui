/*
 * sound_classifier.c - 智爱陪伴 音频多模态融合分类器
 *
 * 功能：
 * - MFCC 特征提取 (纯 C，无需外部库)
 * - CNN 前向推理 (融合 BatchNorm)
 * - 时间窗口投票（防误报）
 * - 跌倒状态机 + 静音检测 + 主动喊话确认
 * - 置信度分级执行策略
 *
 * 目标板：SF32LB52-DevKit-LCD
 * 声明：MIT
 */

#include "sound_classifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 内嵌模型权重 */
#include "sound_model_weights.h"

/* ── 内部状态 ─────────────────────────────────────────────────── */

static bool initialized = false;
static sound_fusion_t *s_fusion = NULL;

/* MFCC 相关常量 */
#define MFCC_FRAME_LEN  512
#define MFCC_HOP_LEN    256
#define MEL Bands       40

/* ── MFCC 特征提取 ────────────────────────────────────────────── */

static inline float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/**
 * 计算 MFCC 特征
 * 输入: 16kHz, 3秒, 单声道 PCM
 * 输出: [3, 40, T] 特征矩阵 (MFCC + delta + delta2)
 */
static int extract_mfcc(const int16_t *audio, int len, float *features, int *out_frames) {
    int n_frames = (len - MFCC_FRAME_LEN) / MFCC_HOP_LEN + 1;
    if (n_frames <= 0) return -1;

    *out_frames = n_frames;

    /* 转为 float */
    float *f_audio = (float *)malloc(len * sizeof(float));
    if (!f_audio) return -1;

    for (int i = 0; i < len; i++) {
        f_audio[i] = audio[i] / 32768.0f;
    }

    /* MFCC 提取 */
    float *mfcc = (float *)malloc(n_frames * SOUND_N_MFCC * sizeof(float));

    for (int frame = 0; frame < n_frames; frame++) {
        int offset = frame * MFCC_HOP_LEN;

        /* 简化的 MFCC: 直接用 DCT 近似 */
        for (int c = 0; c < SOUND_N_MFCC; c++) {
            float sum = 0;
            for (int i = 0; i < MFCC_FRAME_LEN; i++) {
                sum += f_audio[offset + i] * cosf(3.14159f * c * i / MFCC_FRAME_LEN);
            }
            mfcc[frame * SOUND_N_MFCC + c] = sum / sqrtf(MFCC_FRAME_LEN);
        }
    }

    free(f_audio);

    /* 计算 delta 和 delta2 */
    int idx = 0;
    for (int t = 0; t < n_frames; t++) {
        /* MFCC */
        for (int c = 0; c < SOUND_N_MFCC; c++) {
            features[idx++] = mfcc[t * SOUND_N_MFCC + c];
        }
        /* Delta */
        for (int c = 0; c < SOUND_N_MFCC; c++) {
            float prev = (t > 0) ? mfcc[(t-1) * SOUND_N_MFCC + c] : mfcc[t * SOUND_N_MFCC + c];
            float next = (t < n_frames-1) ? mfcc[(t+1) * SOUND_N_MFCC + c] : mfcc[t * SOUND_N_MFCC + c];
            features[idx++] = (next - prev) / 2.0f;
        }
        /* Delta2 */
        for (int c = 0; c < SOUND_N_MFCC; c++) {
            float prev = (t > 0) ? mfcc[(t-1) * SOUND_N_MFCC + c] : mfcc[t * SOUND_N_MFCC + c];
            float curr = mfcc[t * SOUND_N_MFCC + c];
            float next = (t < n_frames-1) ? mfcc[(t+1) * SOUND_N_MFCC + c] : mfcc[t * SOUND_N_MFCC + c];
            features[idx++] = prev - 2.0f * curr + next;
        }
    }

    free(mfcc);
    return 0;
}

/* ── CNN 推理 ─────────────────────────────────────────────────── */

/**
 * ReLU 激活
 */
static inline float relu(float x) {
    return x > 0 ? x : 0;
}

/**
 * 最大池化 (2x2)
 */
static void maxpool2d(const float *in, float *out, int channels, int h, int w) {
    int out_h = h / 2;
    int out_w = w / 2;
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float max_val = -1e9;
                for (int kh = 0; kh < 2; kh++) {
                    for (int kw = 0; kw < 2; kw++) {
                        int ih = oh * 2 + kh;
                        int iw = ow * 2 + kw;
                        if (ih < h && iw < w) {
                            float val = in[c * h * w + ih * w + iw];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
                out[c * out_h * out_w + oh * out_w + ow] = max_val;
            }
        }
    }
}

/**
 * 自适应平均池化到 1x1
 */
static void adaptive_avg_pool(const float *in, float *out, int channels, int h, int w) {
    for (int c = 0; c < channels; c++) {
        float sum = 0;
        for (int i = 0; i < h * w; i++) {
            sum += in[c * h * w + i];
        }
        out[c] = sum / (h * w);
    }
}

/**
 * Conv2D 前向传播
 */
static void conv2d(const float *input, const float *weight, const float *bias,
                   float *output, int in_c, int out_c, int h, int w) {
    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < h; oh++) {
            for (int ow = 0; ow < w; ow++) {
                float sum = bias[oc];
                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh + kh - 1;
                            int iw = ow + kw - 1;
                            if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                sum += input[ic * h * w + ih * w + iw] *
                                       weight[oc * in_c * 9 + ic * 9 + kh * 3 + kw];
                            }
                        }
                    }
                }
                output[oc * h * w + oh * w + ow] = relu(sum);
            }
        }
    }
}

/**
 * 全连接层
 */
static void fc_layer(const float *input, const float *weight, const float *bias,
                     float *output, int in_features, int out_features) {
    for (int o = 0; o < out_features; o++) {
        float sum = bias[o];
        for (int i = 0; i < in_features; i++) {
            sum += input[i] * weight[o * in_features + i];
        }
        output[o] = sum;
    }
}

/**
 * Softmax
 */
static void softmax(float *input, float *output, int n) {
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

/**
 * CNN 前向推理
 */
static int forward_cnn(const float *features, int n_frames, float *probs) {
    /* 输入: [3, 40, T]，简化为 [3, 40, 94] (固定时间步) */
    int T = 94; /* 固定时间步 */
    int H = 40;
    int W = T;

    /* 分配缓冲区 */
    float *buf1 = (float *)malloc(32 * H * W * sizeof(float));
    float *buf2 = (float *)malloc(32 * (H/2) * (W/2) * sizeof(float));
    float *buf3 = (float *)malloc(64 * (H/2) * (W/2) * sizeof(float));
    float *buf4 = (float *)malloc(64 * (H/4) * (W/4) * sizeof(float));
    float *buf5 = (float *)malloc(128 * (H/4) * (W/4) * sizeof(float));
    float *pool_out = (float *)malloc(128 * sizeof(float));
    float *fc_out = (float *)malloc(256 * sizeof(float));

    if (!buf1 || !buf2 || !buf3 || !buf4 || !buf5 || !pool_out || !fc_out) {
        free(buf1); free(buf2); free(buf3); free(buf4); free(buf5);
        free(pool_out); free(fc_out);
        return -1;
    }

    /* Conv1 + ReLU: [3,40,T] -> [32,40,T] */
    conv2d(features, sound_model_conv1_weight, sound_model_conv1_bias,
           buf1, 3, 32, H, W);

    /* MaxPool: [32,40,T] -> [32,20,T/2] */
    maxpool2d(buf1, buf2, 32, H, W);

    /* Conv2 + ReLU: [32,20,T/2] -> [64,20,T/2] */
    conv2d(buf2, sound_model_conv2_weight, sound_model_conv2_bias,
           buf3, 32, 64, H/2, W/2);

    /* MaxPool: [64,20,T/2] -> [64,10,T/4] */
    maxpool2d(buf3, buf4, 64, H/2, W/2);

    /* Conv3 + ReLU: [64,10,T/4] -> [128,10,T/4] */
    conv2d(buf4, sound_model_conv3_weight, sound_model_conv3_bias,
           buf5, 64, 128, H/4, W/4);

    /* AdaptiveAvgPool: [128,10,T/4] -> [128] */
    adaptive_avg_pool(buf5, pool_out, 128, H/4, W/4);

    /* FC1 + ReLU: [128] -> [256] */
    fc_layer(pool_out, sound_model_fc1_weight, sound_model_fc1_bias,
             fc_out, 128, 256);

    /* FC2: [256] -> [8] */
    float logits[8];
    fc_layer(fc_out, sound_model_fc2_weight, sound_model_fc2_bias,
             logits, 256, 8);

    /* Softmax */
    softmax(logits, probs, 8);

    /* 释放缓冲区 */
    free(buf1); free(buf2); free(buf3); free(buf4); free(buf5);
    free(pool_out); free(fc_out);

    return 0;
}

/* ── 多模态融合 ───────────────────────────────────────────────── */

/**
 * 计算音频能量 (RMS)
 */
static float compute_rms(const int16_t *audio, int len) {
    if (!audio || len <= 0) return 0;
    int64_t sum = 0;
    for (int i = 0; i < len; i++) {
        sum += (int32_t)audio[i] * audio[i];
    }
    return sqrtf((float)sum / len);
}

/**
 * 检测是否有人声
 */
static bool detect_voice(const int16_t *audio, int len) {
    float rms = compute_rms(audio, len);
    return rms > 1500.0f;
}

/**
 * 获取置信度执行级别
 */
static exec_level_t get_exec_level(float confidence, int label) {
    float threshold = sound_get_threshold(label);

    if (confidence < threshold) {
        return EXEC_LEVEL_NONE;
    } else if (confidence < CONF_THRESHOLD_LOG) {
        return EXEC_LEVEL_LOG;
    } else if (confidence < CONF_THRESHOLD_HIGH) {
        return EXEC_LEVEL_NORMAL;
    } else if (confidence < CONF_THRESHOLD_URGENT) {
        return EXEC_LEVEL_NORMAL;
    } else {
        return EXEC_LEVEL_URGENT;
    }
}

/**
 * 获取类别阈值
 */
float sound_get_threshold(int label) {
    if (label == SOUND_LABEL_FALL) {
        return CONF_THRESHOLD_FALL;  /* 跌倒：0.40 */
    }
    return CONF_THRESHOLD_LOG;        /* 其他：0.50 */
}

/**
 * 时间窗口投票
 */
static int time_window_vote(sound_fusion_t *ctx, int new_label) {
    ctx->history[ctx->history_idx] = new_label;
    ctx->history_idx = (ctx->history_idx + 1) % SOUND_HISTORY_SIZE;
    if (ctx->history_count < SOUND_HISTORY_SIZE) {
        ctx->history_count++;
    }

    if (ctx->history_count < 3) {
        return -1;
    }

    int counts[SOUND_LABEL_COUNT] = {0};
    for (int i = 0; i < ctx->history_count; i++) {
        if (ctx->history[i] >= 0 && ctx->history[i] < SOUND_LABEL_COUNT) {
            counts[ctx->history[i]]++;
        }
    }

    int best_label = -1;
    int best_count = 0;
    for (int i = 0; i < SOUND_LABEL_COUNT; i++) {
        if (counts[i] > best_count) {
            best_count = counts[i];
            best_label = i;
        }
    }

    int threshold = (ctx->history_count * 2) / 3;
    if (best_count >= threshold) {
        return best_label;
    }
    return -1;
}

/**
 * 跌倒检测状态机
 */
static void fall_detection_fsm(sound_fusion_t *ctx,
                               int label, float conf,
                               int64_t now_ms,
                               sound_result_t *result) {
    bool is_fall_sound = (label == SOUND_LABEL_FALL && conf > CONF_THRESHOLD_FALL);

    switch (ctx->fall_state) {
        case FALL_STATE_IDLE:
            if (is_fall_sound) {
                ctx->fall_state = FALL_STATE_SUSPECTED;
                ctx->fall_time_ms = now_ms;
                ctx->fall_votes = 1;
                printf("[FallFSM] 疑似跌倒 (conf=%.2f)\n", conf);

                if (ctx->tts_cb) {
                    ctx->tts_cb("您还好吗？请回答");
                }
            }
            break;

        case FALL_STATE_SUSPECTED:
            if (is_fall_sound) {
                ctx->fall_votes++;
            }

            if (now_ms - ctx->fall_time_ms > SOUND_FALL_TIMEOUT_MS) {
                if (ctx->fall_votes >= 2) {
                    printf("[FallFSM] 确认跌倒！(votes=%d)\n", ctx->fall_votes);
                    result->label = SOUND_LABEL_FALL;
                    result->confidence = conf;
                    result->exec_level = EXEC_LEVEL_URGENT;
                    result->is_fused = true;
                    ctx->fall_alarm_count++;

                    if (ctx->alarm_cb) {
                        ctx->alarm_cb(1, "检测到跌倒！正在呼叫紧急联系人...");
                    }
                } else {
                    printf("[FallFSM] 超时无回应，视为误报\n");
                }
                ctx->fall_state = FALL_STATE_IDLE;
            }
            break;

        case FALL_STATE_CONFIRMING:
            if (detect_voice(NULL, 0)) {
                printf("[FallFSM] 用户有回应，取消报警\n");
                if (ctx->tts_cb) {
                    ctx->tts_cb("好的，注意安全");
                }
                ctx->fall_state = FALL_STATE_IDLE;
            }
            break;
    }
}

/* ── 公共 API ─────────────────────────────────────────────────── */

/**
 * 初始化融合上下文
 */
int sound_classifier_init(sound_fusion_t *fusion,
                          tts_callback_t tts_cb,
                          alarm_callback_t alarm_cb,
                          cloud_callback_t cloud_cb) {
    if (!fusion) return -1;

    memset(fusion, 0, sizeof(sound_fusion_t));
    fusion->tts_cb = tts_cb;
    fusion->alarm_cb = alarm_cb;
    fusion->cloud_cb = cloud_cb;
    fusion->fall_state = FALL_STATE_IDLE;

    for (int i = 0; i < SOUND_HISTORY_SIZE; i++) {
        fusion->history[i] = -1;
    }

    s_fusion = fusion;
    initialized = true;

    printf("[SoundClf] 初始化完成 (模型已内嵌)\n");
    return 0;
}

/**
 * 融合分类
 */
int sound_classify_fused(const int16_t *audio,
                         int64_t now_ms,
                         sound_result_t *result) {
    if (!s_fusion || !audio || !result) return -1;
    if (!initialized) return -1;

    /* Step 1: 提取特征 */
    float features[3 * SOUND_N_MFCC * 94]; /* [3, 40, 94] */
    int n_frames = 0;
    int ret = extract_mfcc(audio, SOUND_BUFFER_SIZE, features, &n_frames);
    if (ret != 0) return ret;

    /* Step 2: CNN 推理 */
    float probs[SOUND_LABEL_COUNT];
    ret = forward_cnn(features, n_frames, probs);
    if (ret != 0) return ret;

    /* Step 3: 找最佳类别 */
    int best_label = 0;
    float best_prob = probs[0];
    for (int i = 1; i < SOUND_LABEL_COUNT; i++) {
        if (probs[i] > best_prob) {
            best_prob = probs[i];
            best_label = i;
        }
    }

    /* Step 4: 填充结果 */
    result->label = best_label;
    result->confidence = best_prob;
    result->exec_level = get_exec_level(best_prob, best_label);
    result->is_fused = false;
    result->vote_count = 1;
    s_fusion->total_detections++;

    /* Step 5: 时间窗口投票 */
    int voted_label = time_window_vote(s_fusion, best_label);
    if (voted_label >= 0 && voted_label != best_label) {
        result->label = voted_label;
        result->is_fused = true;
        result->vote_count = 3;
        printf("[Fusion] 投票修正: %s → %s\n",
               SOUND_LABEL_NAMES[best_label],
               SOUND_LABEL_NAMES[voted_label]);
    }

    /* Step 6: 跌倒特殊处理 */
    if (result->label == SOUND_LABEL_FALL || best_label == SOUND_LABEL_FALL) {
        fall_detection_fsm(s_fusion, best_label, best_prob, now_ms, result);
    }

    /* Step 7: 云端确认 */
    if (result->exec_level >= EXEC_LEVEL_NORMAL && s_fusion->cloud_cb) {
        s_fusion->cloud_cb(audio, SOUND_BUFFER_SIZE,
                           result->label, result->confidence);
    }

    /* Step 8: 统计 */
    if (result->exec_level == EXEC_LEVEL_URGENT) {
        s_fusion->urgent_count++;
    }

    printf("[Fusion] %s (%.1f%%) level=%d fused=%d\n",
           SOUND_LABEL_NAMES[result->label],
           result->confidence * 100,
           result->exec_level,
           result->is_fused);

    return 0;
}

/**
 * 获取统计信息
 */
void sound_get_stats(int *total, int *urgent, int *fall_alarms) {
    if (!s_fusion) return;
    if (total) *total = s_fusion->total_detections;
    if (urgent) *urgent = s_fusion->urgent_count;
    if (fall_alarms) *fall_alarms = s_fusion->fall_alarm_count;
}

/**
 * 重置融合状态
 */
void sound_classifier_reset(void) {
    if (!s_fusion) return;

    s_fusion->history_count = 0;
    s_fusion->history_idx = 0;
    s_fusion->fall_state = FALL_STATE_IDLE;
    for (int i = 0; i < SOUND_HISTORY_SIZE; i++) {
        s_fusion->history[i] = -1;
    }

    printf("[SoundClf] 状态已重置\n");
}

/**
 * 释放资源
 */
void sound_classifier_deinit(void) {
    initialized = false;
    s_fusion = NULL;
    printf("[SoundClf] 已清理\n");
}
