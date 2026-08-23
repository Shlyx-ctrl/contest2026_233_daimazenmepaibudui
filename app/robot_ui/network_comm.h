/**
 * network_comm.h - 网络通信模块头文件
 * WiFi 连接、MQTT 通信、云端交互
 */

#ifndef __NETWORK_COMM_H
#define __NETWORK_COMM_H

#include <stdint.h>
#include <stdbool.h>

/* ==================== WiFi 配置 ==================== */
typedef struct {
    char ssid[64];          // WiFi 名称
    char password[64];      // WiFi 密码
    bool connected;         // 连接状态
    int rssi;               // 信号强度
} wifi_config_t;

/* ==================== MQTT 配置 ==================== */
typedef struct {
    char broker[128];       // MQTT 服务器地址
    uint16_t port;          // 端口号
    char client_id[64];     // 客户端 ID
    char username[64];      // 用户名
    char password[64];      // 密码
    bool connected;         // 连接状态
} mqtt_config_t;

/* ==================== 设备状态 ==================== */
typedef struct {
    float temperature;      // 温度
    float humidity;         // 湿度
    int battery_level;      // 电量
    char status[32];        // 状态
    bool alarm_active;      // 报警状态
} device_status_t;

/* ==================== 消息类型 ==================== */
typedef enum {
    MSG_TYPE_STATUS,        // 状态上报
    MSG_TYPE_ALARM,         // 报警消息
    MSG_TYPE_REMINDER,      // 提醒消息
    MSG_TYPE_COMMAND,       // 控制命令
    MSG_TYPE_HEARTBEAT,     // 心跳
    MSG_TYPE_MAX
} msg_type_t;

/* ==================== 回调函数类型 ==================== */
typedef void (*mqtt_msg_callback_t)(const char *topic, const char *payload);
typedef void (*wifi_status_callback_t)(bool connected);
typedef void (*alarm_callback_t)(const char *alarm_type, const char *details);

/* ==================== 初始化函数 ==================== */
int network_comm_init(void);
void network_comm_deinit(void);

/* ==================== WiFi 函数 ==================== */
int wifi_connect(const char *ssid, const char *password);
int wifi_disconnect(void);
bool wifi_is_connected(void);
int wifi_get_rssi(void);

/* ==================== MQTT 函数 ==================== */
int mqtt_connect(const char *broker, uint16_t port, 
                const char *client_id, const char *username, const char *password);
int mqtt_disconnect(void);
bool mqtt_is_connected(void);
int mqtt_subscribe(const char *topic, int qos);
int mqtt_unsubscribe(const char *topic);
int mqtt_publish(const char *topic, const char *payload, int qos, bool retain);

/* ==================== 数据上报函数 ==================== */
int report_device_status(const device_status_t *status);
int report_alarm(const char *alarm_type, const char *details);
int report_heartbeat(void);

/* ==================== 远程控制函数 ==================== */
int send_command_response(const char *cmd_id, bool success, const char *message);

/* ==================== 回调注册函数 ==================== */
void network_set_mqtt_callback(mqtt_msg_callback_t callback);
void network_set_wifi_callback(wifi_status_callback_t callback);
void network_set_alarm_callback(alarm_callback_t callback);

#endif /* __NETWORK_COMM_H */
