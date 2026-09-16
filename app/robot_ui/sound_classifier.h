/*
 * sound_classifier.h - 智爱陪伴 音频分类器接口
 *
 * 目标板：SF32LB52-DevKit-LCD
 */

#ifndef SOUND_CLASSIFIER_H
#define SOUND_CLASSIFIER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*═══════════════════════════════════════════════════════════════════
 * 音频参数
 *═══════════════════════════════════════════════════════════════════*/
#define SOUND_SAMPLE_RATE       16000
#define SOUND_DURATION_SEC      3
#define SOUND_BUFFER_SIZE       (SOUND_SAMPLE_RATE * SOUND_DURATION_SEC)  /* 48000 */

/*═══════════════════════════════════════════════════════════════════
 * MFCC 特征参数
 *═══════════════════════════════════════════════════════════════════*/
#define SOUND_N_MFCC            40      /* MFCC 系数数量 */
#define SOUND_N_FFT             512     /* FFT 窗口大小 */
#define SOUND_HOP_LENGTH        256     /* 帧移 */
#define SOUND_N_CHANNELS        3       /* MFCC + delta + delta2 */

/*═══════════════════════════════════════════════════════════════════
 * CNN 模型参数 (与 sound_model_weights.h 对齐)
 *═══════════════════════════════════════════════════════════════════*/
#define SOUND_MODEL_INPUT_C     3       /* 输入通道 */
#define SOUND_MODEL_INPUT_H     40      /* MFCC 系数维度 */
#define SOUND_MODEL_INPUT_W     94      /* 时间步 */
#define SOUND_MODEL_CONV1_OUT   32
#define SOUND_MODEL_CONV2_OUT   64
#define SOUND_MODEL_CONV3_OUT   128
#define SOUND_MODEL_FC1_OUT     256
#define SOUND_MODEL_NUM_CLASSES 8

/*═══════════════════════════════════════════════════════════════════
 * 融合参数
 *═══════════════════════════════════════════════════════════════════*/
#define SOUND_HISTORY_SIZE      5
#define SOUND_FALL_TIMEOUT_MS   5000

/*═══════════════════════════════════════════════════════════════════
 * 置信度阈值
 *═══════════════════════════════════════════════════════════════════*/
#define CONF_THRESHOLD_URGENT   0.85f
#define CONF_THRESHOLD_HIGH     0.65f
#define CONF_THRESHOLD_LOG      0.50f
#define CONF_THRESHOLD_FALL     0.40f

/*═══════════════════════════════════════════════════════════════════
 * 标签定义
 *═══════════════════════════════════════════════════════════════════*/
#define SOUND_LABEL_COUGH       0
#define SOUND_LABEL_DOOR        1
#define SOUND_LABEL_FALL        2
#define SOUND_LABEL_FOOTSTEPS   3
#define SOUND_LABEL_NOISE       4
#define SOUND_LABEL_OTHER       5
#define SOUND_LABEL_SCREAM      6
#define SOUND_LABEL_WATER       7
#define SOUND_LABEL_COUNT       8

static const char *SOUND_LABEL_NAMES[] = {
    "咳嗽", "开门", "跌倒", "脚步", "噪音", "其他", "尖叫", "水流"
};

/*═══════════════════════════════════════════════════════════════════
 * 数据类型
 *═══════════════════════════════════════════════════════════════════*/
typedef enum {
    EXEC_LEVEL_NONE = 0,
    EXEC_LEVEL_LOG,
    EXEC_LEVEL_NORMAL,
    EXEC_LEVEL_URGENT
} exec_level_t;

typedef enum {
    FALL_STATE_IDLE = 0,
    FALL_STATE_SUSPECTED,
    FALL_STATE_CONFIRMING
} fall_state_t;

typedef struct {
    int label;
    float confidence;
    exec_level_t exec_level;
    bool is_fused;
    int vote_count;
} sound_result_t;

typedef void (*tts_callback_t)(const char *text);
typedef void (*alarm_callback_t)(int level, const char *msg);
typedef void (*cloud_callback_t)(const int16_t *audio, int len, int label, float conf);

typedef struct {
    int history[SOUND_HISTORY_SIZE];
    int history_count;
    int history_idx;
    fall_state_t fall_state;
    int64_t fall_time_ms;
    int fall_votes;
    tts_callback_t tts_cb;
    alarm_callback_t alarm_cb;
    cloud_callback_t cloud_cb;
    int total_detections;
    int urgent_count;
    int fall_alarm_count;
} sound_fusion_t;

/*═══════════════════════════════════════════════════════════════════
 * API
 *═══════════════════════════════════════════════════════════════════*/
int sound_classifier_init(sound_fusion_t *fusion,
                          tts_callback_t tts_cb,
                          alarm_callback_t alarm_cb,
                          cloud_callback_t cloud_cb);

int sound_classify_fused(const int16_t *audio,
                         int64_t now_ms,
                         sound_result_t *result);

float sound_get_threshold(int label);
void sound_get_stats(int *total, int *urgent, int *fall_alarms);
void sound_classifier_reset(void);
void sound_classifier_deinit(void);

#ifdef __cplusplus
}
#endif

#endif /* SOUND_CLASSIFIER_H */
