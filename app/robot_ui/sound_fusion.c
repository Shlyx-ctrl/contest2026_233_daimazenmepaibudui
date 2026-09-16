/*
 * sound_fusion.c - 声音融合决策
 */

#include "sound_fusion.h"
#include <stdio.h>
#include <string.h>

/* 置信度阈值 */
static float get_threshold(int label) {
    return (label == SOUND_LABEL_FALL) ? CONF_THRESHOLD_FALL : CONF_THRESHOLD_LOG;
}

static exec_level_t get_exec_level(float conf, int label) {
    float th = get_threshold(label);
    if (conf < th) return EXEC_NONE;
    if (conf < CONF_THRESHOLD_HIGH) return EXEC_NORMAL;
    if (conf < CONF_THRESHOLD_URGENT) return EXEC_NORMAL;
    return EXEC_URGENT;
}

/* 时间窗口投票 */
static int time_window_vote(sound_fusion_t *ctx, int new_label) {
    ctx->history[ctx->history_idx] = new_label;
    ctx->history_idx = (ctx->history_idx + 1) % FUSION_HISTORY_SIZE;
    if (ctx->history_count < FUSION_HISTORY_SIZE) ctx->history_count++;
    if (ctx->history_count < 3) return -1;

    int counts[SOUND_MODEL_NUM_CLASSES] = {0};
    for (int i = 0; i < ctx->history_count; i++) {
        if (ctx->history[i] >= 0 && ctx->history[i] < SOUND_MODEL_NUM_CLASSES)
            counts[ctx->history[i]]++;
    }

    int best = -1, best_c = 0;
    for (int i = 0; i < SOUND_MODEL_NUM_CLASSES; i++) {
        if (counts[i] > best_c) { best_c = counts[i]; best = i; }
    }
    return (best_c >= (ctx->history_count * 2) / 3) ? best : -1;
}

/* 跌倒状态机 */
static void fall_fsm(sound_fusion_t *ctx, int label, float conf,
                     int64_t now_ms, fusion_result_t *out) {
    bool is_fall = (label == SOUND_LABEL_FALL && conf > CONF_THRESHOLD_FALL);

    switch (ctx->fall_state) {
    case FALL_IDLE:
        if (is_fall) {
            ctx->fall_state = FALL_SUSPECTED;
            ctx->fall_time_ms = now_ms;
            ctx->fall_votes = 1;
            /* 标记需要 TTS（调用者负责异步请求） */
            out->need_tts = true;
            out->tts_text = "您还好吗？请回答";
        }
        break;

    case FALL_SUSPECTED:
        if (is_fall) ctx->fall_votes++;

        if (now_ms - ctx->fall_time_ms > FUSION_FALL_TIMEOUT_MS) {
            if (ctx->fall_votes >= 2) {
                /* 确认跌倒 */
                out->need_alarm = true;
                out->alarm_msg = "检测到跌倒！正在呼叫紧急联系人...";
                ctx->fall_alarm_count++;
            }
            ctx->fall_state = FALL_IDLE;
        }
        break;

    case FALL_CONFIRMING:
        /* 用户回应后，由外部调用 sound_fusion_user_responded() */
        break;
    }
}

/* ═══════════════════════════════════════════════════════════════ */

void sound_fusion_init(sound_fusion_t *ctx) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(sound_fusion_t));
    for (int i = 0; i < FUSION_HISTORY_SIZE; i++) ctx->history[i] = -1;
}

void sound_fusion_reset(sound_fusion_t *ctx) {
    if (!ctx) return;
    ctx->history_count = 0;
    ctx->history_idx = 0;
    ctx->fall_state = FALL_IDLE;
    for (int i = 0; i < FUSION_HISTORY_SIZE; i++) ctx->history[i] = -1;
}

void sound_fusion_decide(sound_fusion_t *ctx,
                         const sound_result_t *result,
                         int64_t now_ms,
                         fusion_result_t *out) {
    if (!ctx || !result || !out) return;

    memset(out, 0, sizeof(fusion_result_t));
    out->label = result->label;
    out->confidence = result->confidence;
    out->exec_level = get_exec_level(result->confidence, result->label);

    ctx->total_detections++;

    /* 时间窗口投票 */
    int voted = time_window_vote(ctx, result->label);
    if (voted >= 0 && voted != result->label) {
        out->label = voted;
        out->is_fused = true;
        out->vote_count = 3;
    }

    /* 跌倒特殊处理 */
    if (out->label == SOUND_LABEL_FALL || result->label == SOUND_LABEL_FALL) {
        fall_fsm(ctx, result->label, result->confidence, now_ms, out);
    }

    /* 统计 */
    if (out->exec_level == EXEC_URGENT) ctx->urgent_count++;
}
