/*
 * sound_fusion.h - 声音融合决策
 */

#ifndef SOUND_FUSION_H
#define SOUND_FUSION_H

#include "sound_classifier.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FUSION_HISTORY_SIZE      5
#define FUSION_FALL_TIMEOUT_MS   5000

#define CONF_THRESHOLD_URGENT   0.85f
#define CONF_THRESHOLD_HIGH     0.65f
#define CONF_THRESHOLD_LOG      0.50f
#define CONF_THRESHOLD_FALL     0.40f

typedef enum { EXEC_NONE = 0, EXEC_LOG, EXEC_NORMAL, EXEC_URGENT } exec_level_t;
typedef enum { FALL_IDLE = 0, FALL_SUSPECTED, FALL_CONFIRMING } fall_state_t;

typedef struct {
    int label;
    float confidence;
    exec_level_t exec_level;
    bool is_fused;
    int vote_count;
    bool need_tts;
    int tts_id;             /* 预缓存的语音 ID */
    bool need_alarm;
    const char *alarm_msg;
} fusion_result_t;

typedef struct {
    int history[FUSION_HISTORY_SIZE];
    int history_count;
    int history_idx;
    fall_state_t fall_state;
    int64_t fall_time_ms;
    int fall_votes;
    int total_detections;
    int urgent_count;
    int fall_alarm_count;
} sound_fusion_t;

void sound_fusion_init(sound_fusion_t *ctx);
void sound_fusion_reset(sound_fusion_t *ctx);
void sound_fusion_decide(sound_fusion_t *ctx,
                         const sound_result_t *result,
                         int64_t now_ms,
                         fusion_result_t *out);

#ifdef __cplusplus
}
#endif

#endif
