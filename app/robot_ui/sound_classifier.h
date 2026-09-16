/*
 * sound_classifier.h - 智爱陪伴 音频多模态融合分类器
 *
 * 功能：纯 C 实现的音频分类 + 多模态融合推理
 * - MFCC 特征提取 (纯 C，无需外部库)
 * - FC 层前向推理 (float 权重，无需 float16)
 * - 时间窗口投票（防误报）
 * - 跌倒状态机 + 静音检测 + 主动喊话确认
 * - 置信度分级执行策略
 *
 * 目标板：SF32LB52-DevKit-LCD
 * 声明：MIT
 */

#ifndef SOUND_CLASSIFIER_H
#define SOUND_CLASSIFIER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * 参数定义
 * ==========================================================================*/
#define SOUND_SAMPLE_RATE       16000
#define SOUND_DURATION_SEC      3
#define SOUND_BUFFER_SIZE       (SOUND_SAMPLE_RATE * SOUND_DURATION_SEC)
#define SOUND_MFCC_COEFFS       40
#define SOUND_N_MFCC            40      /* MFCC 系数数量 */
#define SOUND_N_FFT             512     /* FFT 窗口大小 */
#define SOUND_N_CHANNELS        3       /* MFCC + delta + delta2 */
#define SOUND_FEATURE_CHANNELS  3   /* MFCC + delta + delta2 */
#define SOUND_FRAME_SIZE        512
#define SOUND_HOP_SIZE          256
#define SOUND_N_MELS            40

/* 融合参数 */
#define SOUND_HISTORY_SIZE      5       /* 时间窗口大小 */
#define SOUND_FALL_TIMEOUT_MS   5000    /* 跌倒等待回应超时 */
#define SOUND_SILENCE_THRES     500     /* 静音能量阈值 */
#define SOUND_SILENCE_FRAMES    10      /* 连续静音帧数阈值 */

/* 置信度阈值 */
#define CONF_THRESHOLD_URGENT   0.85f   /* 紧急执行 */
#define CONF_THRESHOLD_HIGH     0.65f   /* 普通执行 */
#define CONF_THRESHOLD_LOG      0.50f   /* 仅记录 */
#define CONF_THRESHOLD_FALL     0.40f   /* 跌倒低阈值（宁可误报不可漏报） */

/*============================================================================
 * 标签定义
 * ==========================================================================*/
#define SOUND_LABEL_COUGH       0
#define SOUND_LABEL_DOOR        1
#define SOUND_LABEL_FALL        2
#define SOUND_LABEL_FOOTSTEPS   3
#define SOUND_LABEL_NOISE       4
#define SOUND_LABEL_OTHER       5
#define SOUND_LABEL_SCREAM      6
#define SOUND_LABEL_WATER       7
#define SOUND_LABEL_COUNT       8

/* 标签名称（调试用） */
static const char *SOUND_LABEL_NAMES[] = {
    "咳嗽", "开门", "跌倒", "脚步", "噪音", "其他", "尖叫", "水流"
};

/*============================================================================
 * 置信度执行级别
 * ==========================================================================*/
typedef enum {
    EXEC_LEVEL_NONE = 0,     /* < 0.50: 不执行，仅记录 */
    EXEC_LEVEL_LOG,          /* 0.50-0.65: 仅日志记录 */
    EXEC_LEVEL_NORMAL,       /* 0.65-0.85: 正常执行 */
    EXEC_LEVEL_URGENT        /* > 0.85: 紧急执行 */
} exec_level_t;

/*============================================================================
 * 跌倒检测状态机
 * ==========================================================================*/
typedef enum {
    FALL_STATE_IDLE = 0,     /* 正常监听 */
    FALL_STATE_SUSPECTED,    /* 检测到疑似跌倒声 */
    FALL_STATE_CONFIRMING    /* 等待用户回应 */
} fall_state_t;

/*============================================================================
 * 检测结果
 * ==========================================================================*/
typedef struct {
    int label;               /* 预测标签 */
    float confidence;        /* 置信度 0-1 */
    exec_level_t exec_level; /* 执行级别 */
    bool is_fused;           /* 是否经过融合 */
    int vote_count;          /* 投票一致性计数 */
} sound_result_t;

/*============================================================================
 * 融合回调函数类型
 * ==========================================================================*/
/* TTS 播报回调：播放语音提示 */
typedef void (*tts_callback_t)(const char *text);
/* 紧急报警回调：触发报警 */
typedef void (*alarm_callback_t)(int level, const char *msg);
/* 云端确认回调：发送音频到云端二次确认 */
typedef void (*cloud_callback_t)(const int16_t *audio, int len, int label, float conf);

/*============================================================================
 * 融合上下文（需在初始化时分配）
 * ==========================================================================*/
typedef struct {
    /* 时间窗口投票 */
    int history[SOUND_HISTORY_SIZE];   /* 最近 N 次检测标签 */
    int history_count;                  /* 已填充的帧数 */
    int history_idx;                    /* 写入位置（环形） */

    /* 跌倒状态机 */
    fall_state_t fall_state;
    int64_t fall_time_ms;              /* 疑似跌倒的时间戳 */
    int fall_votes;                    /* 窗口内跌倒计数 */

    /* 静音检测 */
    int silence_frame_count;           /* 连续静音帧计数 */
    bool last_was_silence;

    /* 回调 */
    tts_callback_t tts_cb;
    alarm_callback_t alarm_cb;
    cloud_callback_t cloud_cb;

    /* 统计 */
    int total_detections;
    int urgent_count;
    int fall_alarm_count;
} sound_fusion_t;

/*============================================================================
 * API
 * ==========================================================================*/

/**
 * 初始化分类器 + 融合上下文
 * @param fusion  融合上下文（调用者分配，内部初始化）
 * @param tts_cb  TTS 播报回调（可为 NULL）
 * @param alarm_cb 报警回调（可为 NULL）
 * @param cloud_cb 云端确认回调（可为 NULL）
 * @return 0 成功，-1 失败
 */
int sound_classifier_init(sound_fusion_t *fusion,
                          tts_callback_t tts_cb,
                          alarm_callback_t alarm_cb,
                          cloud_callback_t cloud_cb);

/**
 * 单帧特征分类（不含融合）
 * @param audio     PCM 音频数据（16kHz, 16bit, 单声道, 3秒）
 * @param result    输出：原始分类结果
 * @return 0 成功
 */
int sound_classify_raw(const int16_t *audio, sound_result_t *result);

/**
 * 融合分类（时间窗口投票 + 跌倒状态机）
 * @param audio     PCM 音频数据
 * @param now_ms    当前时间戳（毫秒）
 * @param result    输出：融合后结果
 * @return 0 成功
 */
int sound_classify_fused(const int16_t *audio,
                         int64_t now_ms,
                         sound_result_t *result);

/**
 * 获取当前置信度阈值
 * @param label  类别 ID
 * @return 对应阈值（跌倒类更低）
 */
float sound_get_threshold(int label);

/**
 * 获取统计信息
 */
void sound_get_stats(int *total, int *urgent, int *fall_alarms);

/**
 * 重置融合状态
 */
void sound_classifier_reset(void);

/**
 * 释放资源
 */
void sound_classifier_deinit(void);

#ifdef __cplusplus
}
#endif

#endif /* SOUND_CLASSIFIER_H */
