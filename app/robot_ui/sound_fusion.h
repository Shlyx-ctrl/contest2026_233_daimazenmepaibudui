/*
 * sound_fusion.h - 声音融合决策（状态机 + 投票 + 业务逻辑）
 *
 * 注意：TTS 请求是异步的，不在这个模块里阻塞
 */

#ifndef SOUND_FUSION_H
#define SOUND_FUSION_H

#include "sound_classifier.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 融合参数 */
#define FUSION_HISTORY_SIZE      5
#define FUSION_FALL_TIMEOUT_MS   5000    /* 跌倒等待回应超时 */

/* 置信度阈值 */
#define CONF_THRESHOLD_URGENT   0.85f
#define CONF_THRESHOLD_HIGH     0.65f
#define CONF_THRESHOLD_LOG      0.50f
#define CONF_THRESHOLD_FALL     0.40f

/* 执行级别 */
typedef enum {
    EXEC_NONE = 0,
    EXEC_LOG,
    EXEC_NORMAL,
    EXEC_URGENT
} exec_level_t;

/* 跌倒状态 */
typedef enum {
    FALL_IDLE = 0,
    FALL_SUSPECTED,
    FALL_CONFIRMING
} fall_state_t;

/* 融合结果 */
typedef struct {
    int label;
    float confidence;
    exec_level_t exec_level;
    bool is_fused;
    int vote_count;
    /* 动作指示（由调用者执行） */
    bool need_tts;          /* 需要播放语音 */
    const char *tts_text;   /* TTS 文本（静态字符串，不要 free） */
    bool need_alarm;        /* 需要报警 */
    const char *alarm_msg;  /* 报警信息 */
} fusion_result_t;

/* 回调：TTS 请求（异步，不阻塞） */
typedef void (*tts_request_cb_t)(const char *text, void *user_data);
/* 回调：报警请求 */
typedef void (*alarm_request_cb_t)(int level, const char *msg, void *user_data);

/* 融合上下文 */
typedef struct {
    int history[FUSION_HISTORY_SIZE];
    int history_count;
    int history_idx;
    fall_state_t fall_state;
    int64_t fall_time_ms;
    int fall_votes;
    /* 统计 */
    int total_detections;
    int urgent_count;
    int fall_alarm_count;
} sound_fusion_t;

/* API */
void sound_fusion_init(sound_fusion_t *ctx);
void sound_fusion_reset(sound_fusion_t *ctx);

/**
 * 融合决策
 * @param ctx       融合上下文
 * @param result    分类器结果
 * @param now_ms    当前时间戳（毫秒）
 * @param fusion_out 融合输出（包含动作指示）
 */
void sound_fusion_decide(sound_fusion_t *ctx,
                         const sound_result_t *result,
                         int64_t now_ms,
                         fusion_result_t *fusion_out);

#ifdef __cplusplus
}
#endif

#endif
