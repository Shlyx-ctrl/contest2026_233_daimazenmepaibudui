/**
 * sound_classifier.c — 板端声音分类推理（纯 C, 无 ONNX Runtime）
 *
 * 实现方式：手写定点推理 / 简化特征 + 轻量 MLP
 * 如果 ONNX Runtime for NuttX 可用，可切换到 ONNX 推理。
 *
 * 当前实现：
 *   1. 提取 MFCC 特征（简化版，仅用 cos 变换）
 *   2. 通过预训练权重做前向推理
 *   3. 返回分类结果
 *
 * 模型文件格式（二进制）：
 *   [4B] magic: 0x534F554E ("SOUN")
 *   [4B] version
 *   [4B] num_classes
 *   [4B] num_mfcc
 *   [4B] num_channels (输入通道数, 通常 3)
 *   [W×H×C] conv1_weight  (float32)
 *   [C]     conv1_bias
 *   ... 以此类推
 *
 * 简化方案：由于板端资源有限，我们用 MFCC 特征 + 简单分类器
 */

#include "sound_classifier.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── 模型权重（编译时嵌入）────────────────────────────────────── */
/* 实际部署时由 Python 脚本生成，这里用占位符 */

#ifndef SOUND_MODEL_PATH
#define SOUND_MODEL_PATH "/data/sound_model.bin"
#endif

/* ── 内部状态 ─────────────────────────────────────────────────── */

static bool initialized = false;
static int64_t last_trigger_time[SOUND_LABEL_COUNT] = {0};

/* MFCC 滤波器组系数（预计算） */
static float mel_filters[SOUND_N_MFCC][SOUND_N_FFT / 2 + 1];
static bool filters_computed = false;

/* ── MFCC 特征提取（简化版）────────────────────────────────────── */

/**
 * 频率 → Mel 频率
 */
static inline float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

/**
 * Mel 频率 → Hz
 */
static inline float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/**
 * 初始化 Mel 滤波器组
 */
static void init_mel_filters(void) {
    if (filters_computed) return;

    float low_mel = hz_to_mel(0);
    float high_mel = hz_to_mel(SOUND_SAMPLE_RATE / 2.0f);

    for (int i = 0; i < SOUND_N_MFCC; i++) {
        float center_mel = low_mel + (high_mel - low_mel) * (i + 1) / (SOUND_N_MFCC + 1);
        float center_hz = mel_to_hz(center_mel);
        int center_bin = (int)(center_hz * SOUND_N_FFT / SOUND_SAMPLE_RATE);

        /* 三角滤波器 */
        for (int j = 0; j <= SOUND_N_FFT / 2; j++) {
            float left_mel = hz_to_mel((float)j * SOUND_SAMPLE_RATE / SOUND_N_FFT);
            float c_mel = center_mel;
            float width = (high_mel - low_mel) / (SOUND_N_MFCC + 1);

            if (j >= center_bin - (int)(width * SOUND_N_FFT / SOUND_SAMPLE_RATE) &&
                j <= center_bin) {
                mel_filters[i][j] = (float)(j - (center_bin - (int)(width * SOUND_N_FFT / SOUND_SAMPLE_RATE)))
                                   / (float)((int)(width * SOUND_SAMPLE_RATE / SOUND_N_FFT));
            } else if (j > center_bin &&
                       j <= center_bin + (int)(width * SOUND_N_FFT / SOUND_SAMPLE_RATE)) {
                mel_filters[i][j] = (float)((center_bin + (int)(width * SOUND_SAMPLE_RATE / SOUND_N_FFT)) - j)
                                   / (float)((int)(width * SOUND_SAMPLE_RATE / SOUND_N_FFT));
            } else {
                mel_filters[i][j] = 0.0f;
            }
        }
    }
    filters_computed = true;
}

/**
 * 简化的 FFT（仅计算功率谱）
 * 使用 DFT 直接计算（对于 1024 点够用，板端有 CMSIS-DSP 可加速）
 */
static void compute_power_spectrum(const float *frame, int frame_len, float *power, int n_bins) {
    for (int k = 0; k < n_bins; k++) {
        float real = 0.0f, imag = 0.0f;
        for (int n = 0; n < frame_len; n++) {
            float angle = 2.0f * M_PI * k * n / frame_len;
            real += frame[n] * cosf(angle);
            imag -= frame[n] * sinf(angle);
        }
        power[k] = (real * real + imag * imag) / frame_len;
    }
}

/**
 * 从音频提取 MFCC 特征
 * 输出: [3][SOUND_N_MFCC][T] — MFCC, delta, delta2
 * T = (num_samples - SOUND_N_FFT) / SOUND_HOP_LENGTH + 1
 */
static int extract_mfcc(const float *audio, int num_samples, float *out, int *out_time) {
    init_mel_filters();

    int n_frames = (num_samples - SOUND_N_FFT) / SOUND_HOP_LENGTH + 1;
    if (n_frames <= 0) return -1;
    *out_time = n_frames;

    float frame[SOUND_N_FFT];
    float power[SOUND_N_FFT / 2 + 1];
    float log_mel[SOUND_N_MFCC];
    float mfcc_matrix[SOUND_N_MFCC * 600]; /* 最大 600 帧 */
    int max_frames = sizeof(mfcc_matrix) / sizeof(mfcc_matrix[0]) / SOUND_N_MFCC;
    if (n_frames > max_frames) n_frames = max_frames;

    /* 提取每帧 MFCC */
    for (int t = 0; t < n_frames; t++) {
        int start = t * SOUND_HOP_LENGTH;

        /* 加汉明窗 */
        for (int i = 0; i < SOUND_N_FFT; i++) {
            float w = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (SOUND_N_FFT - 1));
            frame[i] = (start + i < num_samples) ? audio[start + i] * w : 0.0f;
        }

        /* 功率谱 */
        compute_power_spectrum(frame, SOUND_N_FFT, power, SOUND_N_FFT / 2 + 1);

        /* Mel 滤波 + log */
        for (int m = 0; m < SOUND_N_MFCC; m++) {
            float sum = 0.0f;
            for (int k = 0; k <= SOUND_N_FFT / 2; k++) {
                sum += mel_filters[m][k] * power[k];
            }
            log_mel[m] = logf(sum + 1e-10f);
        }

        /* DCT (简化: 直接取前 SOUND_N_MFCC 个系数) */
        for (int m = 0; m < SOUND_N_MFCC; m++) {
            float val = 0.0f;
            for (int n = 0; n < SOUND_N_MFCC; n++) {
                val += log_mel[n] * cosf(M_PI * m * (2 * n + 1) / (2.0f * SOUND_N_MFCC));
            }
            mfcc_matrix[t * SOUND_N_MFCC + m] = val;
        }
    }

    /* 计算 delta 和 delta2 */
    for (int t = 0; t < n_frames; t++) {
        for (int m = 0; m < SOUND_N_MFCC; m++) {
            float cur = mfcc_matrix[t * SOUND_N_MFCC + m];

            /* delta */
            float prev = (t > 0) ? mfcc_matrix[(t - 1) * SOUND_N_MFCC + m] : cur;
            float next = (t < n_frames - 1) ? mfcc_matrix[(t + 1) * SOUND_N_MFCC + m] : cur;
            float delta = (next - prev) / 2.0f;

            /* delta2 */
            float prev2 = (t > 1) ? mfcc_matrix[(t - 2) * SOUND_N_MFCC + m] : prev;
            float next2 = (t < n_frames - 2) ? mfcc_matrix[(t + 2) * SOUND_N_MFCC + m] : next;
            float delta2 = (next2 - 2.0f * cur + prev2) / 4.0f;

            /* 输出: [channel][time][freq] → flatten */
            int idx_mfcc = 0 * SOUND_N_MFCC * n_frames + t * SOUND_N_MFCC + m;
            int idx_delta = 1 * SOUND_N_MFCC * n_frames + t * SOUND_N_MFCC + m;
            int idx_delta2 = 2 * SOUND_N_MFCC * n_frames + t * SOUND_N_MFCC + m;
            out[idx_mfcc] = cur;
            out[idx_delta] = delta;
            out[idx_delta2] = delta2;
        }
    }

    return 0;
}

/* ── 推理引擎（手写前向传播）───────────────────────────────────── */

/* 简化的 CNN：MFCC → 全连接 → softmax */
/* 实际模型由 Python 训练后导出，这里用占位权重 */

/* 模型权重结构 */
typedef struct {
    float *fc1_weight;   /* [256, 3*40*T] */
    float *fc1_bias;     /* [256] */
    float *fc2_weight;   /* [8, 256] */
    float *fc2_bias;     /* [8] */
    int input_dim;
    int hidden_dim;
    int output_dim;
} sound_model_t;

static sound_model_t model = {0};

/**
 * 加载模型权重
 */
static int load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("[SoundClf] 模型文件不存在: %s\n", path);
        return -1;
    }

    uint32_t magic, version, num_classes, num_mfcc, num_channels;
    fread(&magic, 4, 1, f);
    fread(&version, 4, 1, f);
    fread(&num_classes, 4, 1, f);
    fread(&num_mfcc, 4, 1, f);
    fread(&num_channels, 4, 1, f);

    if (magic != 0x534F554E) {
        printf("[SoundClf] 无效模型文件 (magic=0x%08X)\n", magic);
        fclose(f);
        return -1;
    }

    printf("[SoundClf] 模型: v%d, %d类, %d MFCC, %d通道\n",
           version, num_classes, num_mfcc, num_channels);

    /* 简化：直接加载为全连接模型 */
    model.output_dim = num_classes;
    model.hidden_dim = 256;
    model.input_dim = num_channels * num_mfcc * 94;  /* 固定时间帧 */

    /* 读取权重 */
    model.fc1_weight = malloc(model.hidden_dim * model.input_dim * sizeof(float));
    model.fc1_bias = malloc(model.hidden_dim * sizeof(float));
    model.fc2_weight = malloc(model.output_dim * model.hidden_dim * sizeof(float));
    model.fc2_bias = malloc(model.output_dim * sizeof(float));

    if (!model.fc1_weight || !model.fc1_bias || !model.fc2_weight || !model.fc2_bias) {
        printf("[SoundClf] 内存分配失败\n");
        fclose(f);
        return -1;
    }

    fread(model.fc1_weight, sizeof(float), model.hidden_dim * model.input_dim, f);
    fread(model.fc1_bias, sizeof(float), model.hidden_dim, f);
    fread(model.fc2_weight, sizeof(float), model.output_dim * model.hidden_dim, f);
    fread(model.fc2_bias, sizeof(float), model.output_dim, f);

    fclose(f);
    printf("[SoundClf] 模型加载成功 (%d 参数)\n",
           model.hidden_dim * model.input_dim + model.hidden_dim +
           model.output_dim * model.hidden_dim + model.output_dim);
    return 0;
}

/**
 * 简化的前向推理（全连接 + ReLU + Softmax）
 */
static int forward(const float *input, float *output) {
    float hidden[256];

    /* FC1 + ReLU */
    for (int i = 0; i < model.hidden_dim; i++) {
        float sum = model.fc1_bias[i];
        for (int j = 0; j < model.input_dim; j++) {
            sum += model.fc1_weight[i * model.input_dim + j] * input[j];
        }
        hidden[i] = sum > 0 ? sum : 0;  /* ReLU */
    }

    /* FC2 + Softmax */
    float max_val = -1e10f;
    for (int i = 0; i < model.output_dim; i++) {
        float sum = model.fc2_bias[i];
        for (int j = 0; j < model.hidden_dim; j++) {
            sum += model.fc2_weight[i * model.hidden_dim + j] * hidden[j];
        }
        output[i] = sum;
        if (sum > max_val) max_val = sum;
    }

    /* Softmax */
    float exp_sum = 0;
    for (int i = 0; i < model.output_dim; i++) {
        output[i] = expf(output[i] - max_val);
        exp_sum += output[i];
    }
    for (int i = 0; i < model.output_dim; i++) {
        output[i] /= exp_sum;
    }

    return 0;
}

/* ── 公开接口 ─────────────────────────────────────────────────── */

int sound_classifier_init(void) {
    if (initialized) return 0;

    memset(last_trigger_time, 0, sizeof(last_trigger_time));
    init_mel_filters();

    /* 尝试加载模型 */
    if (load_model(SOUND_MODEL_PATH) == 0) {
        initialized = true;
        printf("[SoundClf] 初始化成功 (模型模式)\n");
        return 0;
    }

    /* 模型文件不存在，用随机权重（演示模式） */
    printf("[SoundClf] ⚠ 模型文件不存在，使用演示模式\n");
    model.output_dim = SOUND_LABEL_COUNT;
    model.hidden_dim = 256;
    model.input_dim = SOUND_N_CHANNELS * SOUND_N_MFCC * 94;
    model.fc1_weight = malloc(model.hidden_dim * model.input_dim * sizeof(float));
    model.fc1_bias = malloc(model.hidden_dim * sizeof(float));
    model.fc2_weight = malloc(model.output_dim * model.hidden_dim * sizeof(float));
    model.fc2_bias = malloc(model.output_dim * sizeof(float));

    if (model.fc1_weight && model.fc1_bias && model.fc2_weight && model.fc2_bias) {
        /* 初始化为小随机值 */
        srand(42);
        for (int i = 0; i < model.hidden_dim * model.input_dim; i++)
            model.fc1_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        for (int i = 0; i < model.hidden_dim; i++)
            model.fc1_bias[i] = 0.0f;
        for (int i = 0; i < model.output_dim * model.hidden_dim; i++)
            model.fc2_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        for (int i = 0; i < model.output_dim; i++)
            model.fc2_bias[i] = 0.0f;
        initialized = true;
        printf("[SoundClf] 初始化成功 (演示模式)\n");
        return 0;
    }

    printf("[SoundClf] 初始化失败\n");
    return -1;
}

int sound_classifier_run(const int16_t *audio, int num_samples, sound_result_t *result) {
    if (!initialized || !result) return -1;
    if (num_samples != SOUND_FRAME_SAMPLES) return -1;

    /* PCM16 → float */
    float f32_audio[SOUND_FRAME_SAMPLES];
    for (int i = 0; i < num_samples; i++) {
        f32_audio[i] = audio[i] / 32768.0f;
    }

    return sound_classifier_run_f32(f32_audio, num_samples, result);
}

int sound_classifier_run_f32(const float *audio, int num_samples, sound_result_t *result) {
    if (!initialized || !result) return -1;

    /* 提取 MFCC */
    int n_frames = 0;
    float *mfcc_features = malloc(SOUND_N_CHANNELS * SOUND_N_MFCC * 600 * sizeof(float));
    if (!mfcc_features) return -1;

    if (extract_mfcc(audio, num_samples, mfcc_features, &n_frames) != 0) {
        free(mfcc_features);
        return -1;
    }

    /* 截取或填充到固定维度 */
    int expected_frames = (SOUND_FRAME_SAMPLES - SOUND_N_FFT) / SOUND_HOP_LENGTH + 1;
    int feature_dim = SOUND_N_CHANNELS * SOUND_N_MFCC * expected_frames;
    float *input = calloc(feature_dim, sizeof(float));
    if (!input) {
        free(mfcc_features);
        return -1;
    }

    int copy_frames = (n_frames < expected_frames) ? n_frames : expected_frames;
    for (int ch = 0; ch < SOUND_N_CHANNELS; ch++) {
        for (int t = 0; t < copy_frames; t++) {
            for (int m = 0; m < SOUND_N_MFCC; m++) {
                int src_idx = ch * SOUND_N_MFCC * n_frames + t * SOUND_N_MFCC + m;
                int dst_idx = ch * SOUND_N_MFCC * expected_frames + t * SOUND_N_MFCC + m;
                input[dst_idx] = mfcc_features[src_idx];
            }
        }
    }
    free(mfcc_features);

    /* 推理 */
    float probs[SOUND_LABEL_COUNT];
    forward(input, probs);
    free(input);

    /* 填充结果 */
    int best_label = 0;
    float best_prob = probs[0];
    for (int i = 1; i < SOUND_LABEL_COUNT; i++) {
        if (probs[i] > best_prob) {
            best_prob = probs[i];
            best_label = i;
        }
    }

    result->label = best_label;
    result->confidence = best_prob;
    memcpy(result->probs, probs, sizeof(probs));
    result->timestamp_ms = 0;  /* 由调用者填充 */

    return 0;
}

bool sound_classifier_debounced(int label) {
    if (label < 0 || label >= SOUND_LABEL_COUNT) return true;

    /* 简单的时间戳比较（需要系统提供 tick） */
    /* 这里用静态变量模拟 */
    static int64_t current_time = 0;
    current_time += 100;  /* 每次调用 +100ms */

    if (current_time - last_trigger_time[label] < SOUND_DEBOUNCE_MS) {
        return true;  /* 仍在去抖期内 */
    }

    last_trigger_time[label] = current_time;
    return false;
}

void sound_classifier_deinit(void) {
    if (model.fc1_weight) free(model.fc1_weight);
    if (model.fc1_bias) free(model.fc1_bias);
    if (model.fc2_weight) free(model.fc2_weight);
    if (model.fc2_bias) free(model.fc2_bias);
    memset(&model, 0, sizeof(model));
    initialized = false;
    printf("[SoundClf] 已清理\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 多模态融合推理
 * ═══════════════════════════════════════════════════════════════════════════ */

/* 静态融合上下文 */
static sound_fusion_t *s_fusion = NULL;

/**
 * 计算音频能量（RMS）
 */
static float compute_rms(const int16_t *audio, int len) {
    int64_t sum = 0;
    for (int i = 0; i < len; i++) {
        sum += (int32_t)audio[i] * audio[i];
    }
    return sqrtf((float)sum / len);
}

/**
 * 检测是否有人声（简单能量 + 频谱质心）
 * 返回 true 表示检测到人声
 */
static bool detect_voice(const int16_t *audio, int len) {
    float rms = compute_rms(audio, len);
    /* 简化：能量超过阈值认为是人声 */
    return rms > 1500.0f;
}

/**
 * 计算置信度执行级别
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
 * 获取类别阈值（跌倒类更低）
 */
float sound_get_threshold(int label) {
    if (label == SOUND_LABEL_FALL) {
        return CONF_THRESHOLD_FALL;  /* 跌倒：0.40，宁可误报 */
    }
    return CONF_THRESHOLD_LOG;        /* 其他：0.50 */
}

/**
 * 时间窗口投票
 * 返回投票后的标签，-1 表示不确定
 */
static int time_window_vote(sound_fusion_t *ctx, int new_label) {
    /* 写入历史 */
    ctx->history[ctx->history_idx] = new_label;
    ctx->history_idx = (ctx->history_idx + 1) % SOUND_HISTORY_SIZE;
    if (ctx->history_count < SOUND_HISTORY_SIZE) {
        ctx->history_count++;
    }

    if (ctx->history_count < 3) {
        return -1;  /* 数据不足，不投票 */
    }

    /* 统计各类别出现次数 */
    int counts[SOUND_LABEL_COUNT] = {0};
    for (int i = 0; i < ctx->history_count; i++) {
        if (ctx->history[i] >= 0 && ctx->history[i] < SOUND_LABEL_COUNT) {
            counts[ctx->history[i]]++;
        }
    }

    /* 找出最多的类别 */
    int best_label = -1;
    int best_count = 0;
    for (int i = 0; i < SOUND_LABEL_COUNT; i++) {
        if (counts[i] > best_count) {
            best_count = counts[i];
            best_label = i;
        }
    }

    /* 至少 2/3 一致才确认 */
    int threshold = (ctx->history_count * 2) / 3;
    if (best_count >= threshold) {
        return best_label;
    }
    return -1;  /* 不确定 */
}

/**
 * 跌倒检测状态机
 * 结合：疑似跌倒声 + 静音 + 主动喊话确认
 */
static void fall_detection_fsm(sound_fusion_t *ctx,
                               int label, float conf,
                               int64_t now_ms,
                               sound_result_t *result) {
    bool is_fall_sound = (label == SOUND_LABEL_FALL && conf > CONF_THRESHOLD_FALL);
    bool is_silence = (compute_rms(NULL, 0) < SOUND_SILENCE_THRES);  /* 简化 */

    switch (ctx->fall_state) {
        case FALL_STATE_IDLE:
            if (is_fall_sound) {
                /* 疑似跌倒 → 进入等待状态 */
                ctx->fall_state = FALL_STATE_SUSPECTED;
                ctx->fall_time_ms = now_ms;
                ctx->fall_votes = 1;
                printf("[FallFSM] 疑似跌倒 (conf=%.2f)，等待确认...\n", conf);

                /* 主动喊话 */
                if (ctx->tts_cb) {
                    ctx->tts_cb("您还好吗？请回答");
                }
            }
            break;

        case FALL_STATE_SUSPECTED:
            /* 等待期间又检测到跌倒声 → 加票 */
            if (is_fall_sound) {
                ctx->fall_votes++;
            }

            /* 检查是否超时 */
            if (now_ms - ctx->fall_time_ms > SOUND_FALL_TIMEOUT_MS) {
                if (ctx->fall_votes >= 2) {
                    /* 多次检测到 → 确认跌倒 */
                    printf("[FallFSM] 确认跌倒！(votes=%d)\n", ctx->fall_votes);
                    result->label = SOUND_LABEL_FALL;
                    result->confidence = conf;
                    result->exec_level = EXEC_LEVEL_URGENT;
                    result->is_fused = true;
                    ctx->fall_alarm_count++;

                    /* 触发紧急报警 */
                    if (ctx->alarm_cb) {
                        ctx->alarm_cb(1, "检测到跌倒！正在呼叫紧急联系人...");
                    }
                } else {
                    /* 单次检测，超时无回应 → 可能误报 */
                    printf("[FallFSM] 超时无回应，视为误报\n");

                    /* 如果超过 10 秒仍无声音 → 再次确认 */
                    if (now_ms - ctx->fall_time_ms > 10000 && !detect_voice(NULL, 0)) {
                        printf("[FallFSM] 长时间无声音，再次报警\n");
                        result->label = SOUND_LABEL_FALL;
                        result->confidence = conf * 0.8f;
                        result->exec_level = EXEC_LEVEL_HIGH;
                        result->is_fused = true;
                        ctx->fall_alarm_count++;

                        if (ctx->alarm_cb) {
                            ctx->alarm_cb(0, "长时间无回应，已通知紧急联系人");
                        }
                    }
                }
                ctx->fall_state = FALL_STATE_IDLE;
            }
            break;

        case FALL_STATE_CONFIRMING:
            /* 用户回应了 → 取消报警 */
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

    /* 初始化历史为 -1（无效） */
    for (int i = 0; i < SOUND_HISTORY_SIZE; i++) {
        fusion->history[i] = -1;
    }

    s_fusion = fusion;

    printf("[SoundClf] 融合模块初始化完成\n");
    return 0;
}

/**
 * 融合分类（主入口）
 */
int sound_classify_fused(const int16_t *audio,
                         int64_t now_ms,
                         sound_result_t *result) {
    if (!s_fusion || !audio || !result) return -1;

    /* Step 1: 原始分类 */
    sound_result_t raw;
    int ret = sound_classify_raw(audio, &raw);
    if (ret != 0) return ret;

    /* Step 2: 置信度分级 */
    result->label = raw.label;
    result->confidence = raw.confidence;
    result->exec_level = get_exec_level(raw.confidence, raw.label);
    result->is_fused = false;
    result->vote_count = 1;
    s_fusion->total_detections++;

    /* Step 3: 时间窗口投票 */
    int voted_label = time_window_vote(s_fusion, raw.label);
    if (voted_label >= 0 && voted_label != raw.label) {
        /* 投票改变了结果 → 更可靠 */
        result->label = voted_label;
        result->is_fused = true;
        result->vote_count = 3;  /* 至少 3 帧一致 */
        printf("[Fusion] 投票修正: %s → %s\n",
               SOUND_LABEL_NAMES[raw.label],
               SOUND_LABEL_NAMES[voted_label]);
    }

    /* Step 4: 跌倒特殊处理 */
    if (result->label == SOUND_LABEL_FALL ||
        raw.label == SOUND_LABEL_FALL) {
        fall_detection_fsm(s_fusion, raw.label, raw.confidence, now_ms, result);
    }

    /* Step 5: 紧急/高置信度 → 云端确认（可选） */
    if (result->exec_level >= EXEC_LEVEL_NORMAL &&
        s_fusion->cloud_cb) {
        printf("[Fusion] 高置信度，发送云端确认\n");
        s_fusion->cloud_cb(audio, SOUND_BUFFER_SIZE,
                           result->label, result->confidence);
    }

    /* Step 6: 统计 */
    if (result->exec_level == EXEC_LEVEL_URGENT) {
        s_fusion->urgent_count++;
    }

    printf("[Fusion] 结果: %s (%.1f%%) level=%d fused=%d votes=%d\n",
           SOUND_LABEL_NAMES[result->label],
           result->confidence * 100,
           result->exec_level,
           result->is_fused,
           result->vote_count);

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
    s_fusion->silence_frame_count = 0;
    for (int i = 0; i < SOUND_HISTORY_SIZE; i++) {
        s_fusion->history[i] = -1;
    }

    printf("[SoundClf] 融合状态已重置\n");
}
