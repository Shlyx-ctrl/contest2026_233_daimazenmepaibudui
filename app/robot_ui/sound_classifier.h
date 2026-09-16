/*
 * sound_classifier.h - 声音分类器（纯分类，不含业务逻辑）
 */

#ifndef SOUND_CLASSIFIER_H
#define SOUND_CLASSIFIER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 音频参数 */
#define SOUND_SAMPLE_RATE       16000
#define SOUND_DURATION_SEC      3
#define SOUND_BUFFER_SIZE       (SOUND_SAMPLE_RATE * SOUND_DURATION_SEC)

/* MFCC 参数 */
#define SOUND_N_MFCC            40
#define SOUND_N_FFT             512
#define SOUND_HOP_LENGTH        256
#define SOUND_N_CHANNELS        3

/* CNN 参数 */
#define SOUND_MODEL_NUM_CLASSES 8

/* 标签 */
#define SOUND_LABEL_COUGH       0
#define SOUND_LABEL_DOOR        1
#define SOUND_LABEL_FALL        2
#define SOUND_LABEL_FOOTSTEPS   3
#define SOUND_LABEL_NOISE       4
#define SOUND_LABEL_OTHER       5
#define SOUND_LABEL_SCREAM      6
#define SOUND_LABEL_WATER       7

static const char *SOUND_LABEL_NAMES[] = {
    "咳嗽", "开门", "跌倒", "脚步", "噪音", "其他", "尖叫", "水流"
};

/* 分类结果 */
typedef struct {
    int label;
    float confidence;
    float probs[SOUND_MODEL_NUM_CLASSES];
} sound_result_t;

/* API */
int sound_classifier_init(void);
int sound_classifier_run(const int16_t *audio, int len, sound_result_t *result);
void sound_classifier_deinit(void);

#ifdef __cplusplus
}
#endif

#endif
