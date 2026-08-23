/**
 * network_comm.c - 网络通信模块实现
 * WiFi 连接、MQTT 通信、云端交互
 */

#include "network_comm.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* NuttX 网络头文件 */
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <nuttx/wireless/wireless.h>

/* cJSON 用于 JSON 解析 */
#include <cjson/cJSON.h>

/* ==================== 全局变量 ==================== */
static wifi_config_t wifi_config = {0};
static mqtt_config_t mqtt_config = {0};
static device_status_t device_status = {0};

/* MQTT 连接 socket */
static int mqtt_socket = -1;

/* 回调函数 */
static mqtt_msg_callback_t mqtt_callback = NULL;
static wifi_status_callback_t wifi_callback = NULL;
static alarm_callback_t alarm_callback = NULL;

/* 心跳定时器 */
static uint32_t last_heartbeat_time = 0;
#define HEARTBEAT_INTERVAL 30000  // 30秒

/* ==================== 内部函数声明 ==================== */
static int create_tcp_socket(const char *host, uint16_t port);
static int mqtt_send_connect(void);
static int mqtt_send_subscribe(const char *topic, int qos);
static int mqtt_send_publish(const char *topic, const char *payload, int qos);
static int mqtt_send_pingreq(void);
static int mqtt_send_disconnect(void);
static int mqtt_parse_packet(void);
static char* create_json_message(msg_type_t type, const void *data);

/* ==================== 初始化网络通信 ==================== */
int network_comm_init(void)
{
    printf("网络通信模块初始化...\n");

    /* 初始化配置 */
    memset(&wifi_config, 0, sizeof(wifi_config));
    memset(&mqtt_config, 0, sizeof(mqtt_config));
    memset(&device_status, 0, sizeof(device_status));

    /* 设置默认 MQTT 服务器 */
    strncpy(mqtt_config.broker, "broker.emqx.io", sizeof(mqtt_config.broker) - 1);
    mqtt_config.port = 1883;
    strncpy(mqtt_config.client_id, "zhi_ai_001", sizeof(mqtt_config.client_id) - 1);

    printf("网络通信模块初始化完成\n");
    return 0;
}

/* ==================== 反初始化 ==================== */
void network_comm_deinit(void)
{
    mqtt_disconnect();
    wifi_disconnect();
    printf("网络通信模块已关闭\n");
}

/* ==================== WiFi 连接 ==================== */
int wifi_connect(const char *ssid, const char *password)
{
    printf("正在连接 WiFi: %s\n", ssid);

    /* 保存配置 */
    strncpy(wifi_config.ssid, ssid, sizeof(wifi_config.ssid) - 1);
    strncpy(wifi_config.password, password, sizeof(wifi_config.password) - 1);

    /* TODO: 调用 NuttX WiFi API 连接
     * 示例代码（需要根据实际硬件调整）：
     *
     * struct wireless_config_s wconfig;
     * strncpy(wconfig.essid, ssid, 32);
     * strncpy(wconfig.password, password, 64);
     * wconfig.auth = WIRELESS_AUTH_WPA2;
     * wconfig.mode = WIRELESS_MODE_STATION;
     *
     * int ret = ioctl(fd, SIOCSIWLCFG, &wconfig);
     */

    /* 模拟连接成功 */
    wifi_config.connected = true;
    wifi_config.rssi = -50;  // 模拟信号强度

    printf("WiFi 连接成功: %s\n", ssid);

    /* 触发回调 */
    if (wifi_callback) {
        wifi_callback(true);
    }

    return 0;
}

/* ==================== WiFi 断开 ==================== */
int wifi_disconnect(void)
{
    wifi_config.connected = false;
    printf("WiFi 已断开\n");

    if (wifi_callback) {
        wifi_callback(false);
    }

    return 0;
}

/* ==================== 检查 WiFi 状态 ==================== */
bool wifi_is_connected(void)
{
    return wifi_config.connected;
}

/* ==================== 获取信号强度 ==================== */
int wifi_get_rssi(void)
{
    return wifi_config.rssi;
}

/* ==================== MQTT 连接 ==================== */
int mqtt_connect(const char *broker, uint16_t port, 
                const char *client_id, const char *username, const char *password)
{
    printf("正在连接 MQTT 服务器: %s:%d\n", broker, port);

    /* 保存配置 */
    strncpy(mqtt_config.broker, broker, sizeof(mqtt_config.broker) - 1);
    mqtt_config.port = port;
    strncpy(mqtt_config.client_id, client_id, sizeof(mqtt_config.client_id) - 1);
    if (username) {
        strncpy(mqtt_config.username, username, sizeof(mqtt_config.username) - 1);
    }
    if (password) {
        strncpy(mqtt_config.password, password, sizeof(mqtt_config.password) - 1);
    }

    /* 创建 TCP 连接 */
    mqtt_socket = create_tcp_socket(broker, port);
    if (mqtt_socket < 0) {
        printf("创建 TCP 连接失败\n");
        return -1;
    }

    /* 发送 MQTT CONNECT 包 */
    if (mqtt_send_connect() < 0) {
        printf("MQTT CONNECT 失败\n");
        close(mqtt_socket);
        mqtt_socket = -1;
        return -1;
    }

    mqtt_config.connected = true;
    printf("MQTT 连接成功\n");

    /* 订阅命令主题 */
    char topic[128];
    snprintf(topic, sizeof(topic), "zhi_ai/%s/command", client_id);
    mqtt_subscribe(topic, 1);

    return 0;
}

/* ==================== MQTT 断开 ==================== */
int mqtt_disconnect(void)
{
    if (mqtt_socket >= 0) {
        mqtt_send_disconnect();
        close(mqtt_socket);
        mqtt_socket = -1;
    }

    mqtt_config.connected = false;
    printf("MQTT 已断开\n");
    return 0;
}

/* ==================== 检查 MQTT 状态 ==================== */
bool mqtt_is_connected(void)
{
    return mqtt_config.connected;
}

/* ==================== MQTT 订阅 ==================== */
int mqtt_subscribe(const char *topic, int qos)
{
    if (!mqtt_config.connected) {
        printf("MQTT 未连接\n");
        return -1;
    }

    printf("订阅主题: %s\n", topic);
    return mqtt_send_subscribe(topic, qos);
}

/* ==================== MQTT 取消订阅 ==================== */
int mqtt_unsubscribe(const char *topic)
{
    if (!mqtt_config.connected) {
        return -1;
    }

    printf("取消订阅: %s\n", topic);
    // TODO: 实现 UNSUBSCRIBE
    return 0;
}

/* ==================== MQTT 发布 ==================== */
int mqtt_publish(const char *topic, const char *payload, int qos, bool retain)
{
    if (!mqtt_config.connected) {
        printf("MQTT 未连接\n");
        return -1;
    }

    printf("发布消息到 %s: %s\n", topic, payload);
    return mqtt_send_publish(topic, payload, qos, retain);
}

/* ==================== 上报设备状态 ==================== */
int report_device_status(const device_status_t *status)
{
    char topic[128];
    snprintf(topic, sizeof(topic), "zhi_ai/%s/status", mqtt_config.client_id);

    char *json = create_json_message(MSG_TYPE_STATUS, status);
    if (!json) {
        return -1;
    }

    int ret = mqtt_publish(topic, json, 0, true);
    free(json);

    return ret;
}

/* ==================== 上报报警 ==================== */
int report_alarm(const char *alarm_type, const char *details)
{
    char topic[128];
    snprintf(topic, sizeof(topic), "zhi_ai/%s/alarm", mqtt_config.client_id);

    /* 构建报警 JSON */
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "type", "alarm");
    cJSON_AddStringToObject(root, "alarm_type", alarm_type);
    cJSON_AddStringToObject(root, "details", details);
    cJSON_AddNumberToObject(root, "timestamp", time(NULL));

    char *json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!json) {
        return -1;
    }

    /* 发布报警消息（QoS 1，确保送达） */
    int ret = mqtt_publish(topic, json, 1, false);
    free(json);

    /* 触发本地报警回调 */
    if (alarm_callback) {
        alarm_callback(alarm_type, details);
    }

    return ret;
}

/* ==================== 上报心跳 ==================== */
int report_heartbeat(void)
{
    char topic[128];
    snprintf(topic, sizeof(topic), "zhi_ai/%s/heartbeat", mqtt_config.client_id);

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "type", "heartbeat");
    cJSON_AddStringToObject(root, "device_id", mqtt_config.client_id);
    cJSON_AddNumberToObject(root, "timestamp", time(NULL));

    char *json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!json) {
        return -1;
    }

    int ret = mqtt_publish(topic, json, 0, false);
    free(json);

    return ret;
}

/* ==================== 发送命令响应 ==================== */
int send_command_response(const char *cmd_id, bool success, const char *message)
{
    char topic[128];
    snprintf(topic, sizeof(topic), "zhi_ai/%s/response", mqtt_config.client_id);

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "cmd_id", cmd_id);
    cJSON_AddBoolToObject(root, "success", success);
    cJSON_AddStringToObject(root, "message", message);

    char *json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    if (!json) {
        return -1;
    }

    int ret = mqtt_publish(topic, json, 1, false);
    free(json);

    return ret;
}

/* ==================== 注册回调函数 ==================== */
void network_set_mqtt_callback(mqtt_msg_callback_t callback)
{
    mqtt_callback = callback;
}

void network_set_wifi_callback(wifi_status_callback_t callback)
{
    wifi_callback = callback;
}

void network_set_alarm_callback(alarm_callback_t callback)
{
    alarm_callback = callback;
}

/* ==================== 内部函数实现 ==================== */

/* 创建 TCP Socket */
static int create_tcp_socket(const char *host, uint16_t port)
{
    int sockfd;
    struct sockaddr_in server_addr;

    /* 创建 socket */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return -1;
    }

    /* 设置服务器地址 */
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    /* 解析主机名或 IP */
    struct hostent *he = gethostbyname(host);
    if (he) {
        memcpy(&server_addr.sin_addr, he->h_addr_list[0], he->h_length);
    } else {
        server_addr.sin_addr.s_addr = inet_addr(host);
    }

    /* 连接服务器 */
    if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect");
        close(sockfd);
        return -1;
    }

    return sockfd;
}

/* 发送 MQTT CONNECT 包 */
static int mqtt_send_connect(void)
{
    /* 简化的 CONNECT 包构建 */
    uint8_t packet[256];
    int pos = 0;

    /* 固定头 */
    packet[pos++] = 0x10;  // CONNECT 类型

    /* 可变头和载荷（简化） */
    uint8_t variable_header[] = {
        0x00, 0x04, 'M', 'Q', 'T', 'T',  // 协议名
        0x04,  // 协议级别
        0xC2,  // 连接标志（用户名+密码+遗嘱+清理会话）
        0x00, 0x3C,  // 保持连接时间 60 秒
    };

    /* 客户端 ID */
    uint8_t client_id_len = strlen(mqtt_config.client_id);
    uint8_t payload[256];
    int payload_pos = 0;

    /* 客户端 ID */
    payload[payload_pos++] = 0x00;
    payload[payload_pos++] = client_id_len;
    memcpy(&payload[payload_pos], mqtt_config.client_id, client_id_len);
    payload_pos += client_id_len;

    /* 遗嘱主题 */
    char will_topic[128];
    snprintf(will_topic, sizeof(will_topic), "zhi_ai/%s/status", mqtt_config.client_id);
    payload[payload_pos++] = 0x00;
    payload[payload_pos++] = strlen(will_topic);
    memcpy(&payload[payload_pos], will_topic, strlen(will_topic));
    payload_pos += strlen(will_topic);

    /* 遗嘱消息 */
    payload[payload_pos++] = 0x00;
    payload[payload_pos++] = 0x02;
    payload[payload_pos++] = 'O';
    payload[payload_pos++] = 'F';
    payload[payload_pos++] = 'F';

    /* 用户名 */
    if (mqtt_config.username[0]) {
        uint8_t username_len = strlen(mqtt_config.username);
        payload[payload_pos++] = 0x00;
        payload[payload_pos++] = username_len;
        memcpy(&payload[payload_pos], mqtt_config.username, username_len);
        payload_pos += username_len;
    }

    /* 密码 */
    if (mqtt_config.password[0]) {
        uint8_t password_len = strlen(mqtt_config.password);
        payload[payload_pos++] = 0x00;
        payload[payload_pos++] = password_len;
        memcpy(&payload[payload_pos], mqtt_config.password, password_len);
        payload_pos += password_len;
    }

    /* 剩余长度 */
    int remaining = sizeof(variable_header) + payload_pos;
    packet[pos++] = remaining;

    /* 复制可变头和载荷 */
    memcpy(&packet[pos], variable_header, sizeof(variable_header));
    pos += sizeof(variable_header);
    memcpy(&packet[pos], payload, payload_pos);
    pos += payload_pos;

    /* 发送 */
    return send(mqtt_socket, packet, pos, 0);
}

/* 发送 MQTT SUBSCRIBE 包 */
static int mqtt_send_subscribe(const char *topic, int qos)
{
    uint8_t packet[256];
    int pos = 0;

    /* 固定头 */
    packet[pos++] = 0x82;  // SUBSCRIBE 类型

    /* 剩余长度 */
    int topic_len = strlen(topic);
    packet[pos++] = 5 + topic_len;  // 2(包ID) + 2(topic长度) + 1(topic) + topic

    /* 包 ID */
    packet[pos++] = 0x00;
    packet[pos++] = 0x01;

    /* 主题过滤器 */
    packet[pos++] = 0x00;
    packet[pos++] = topic_len;
    memcpy(&packet[pos], topic, topic_len);
    pos += topic_len;

    /* QoS */
    packet[pos++] = qos;

    return send(mqtt_socket, packet, pos, 0);
}

/* 发送 MQTT PUBLISH 包 */
static int mqtt_send_publish(const char *topic, const char *payload, int qos, bool retain)
{
    uint8_t packet[1024];
    int pos = 0;

    /* 固定头 */
    uint8_t type = 0x30;  // PUBLISH
    if (qos == 1) type |= 0x02;
    if (qos == 2) type |= 0x04;
    if (retain) type |= 0x01;
    packet[pos++] = type;

    /* 剩余长度 */
    int topic_len = strlen(topic);
    int payload_len = strlen(payload);
    int remaining = 2 + topic_len + payload_len;
    if (qos > 0) remaining += 2;  // 包 ID
    packet[pos++] = remaining;

    /* 主题名 */
    packet[pos++] = 0x00;
    packet[pos++] = topic_len;
    memcpy(&packet[pos], topic, topic_len);
    pos += topic_len;

    /* 包 ID（QoS 1 或 2 时） */
    if (qos > 0) {
        static uint16_t packet_id = 0;
        packet_id++;
        packet[pos++] = (packet_id >> 8) & 0xFF;
        packet[pos++] = packet_id & 0xFF;
    }

    /* 载荷 */
    memcpy(&packet[pos], payload, payload_len);
    pos += payload_len;

    return send(mqtt_socket, packet, pos, 0);
}

/* 发送 MQTT PINGREQ */
static int mqtt_send_pingreq(void)
{
    uint8_t packet[2] = {0xC0, 0x00};
    return send(mqtt_socket, packet, 2, 0);
}

/* 发送 MQTT DISCONNECT */
static int mqtt_send_disconnect(void)
{
    uint8_t packet[2] = {0xE0, 0x00};
    return send(mqtt_socket, packet, 2, 0);
}

/* 解析 MQTT 数据包 */
static int mqtt_parse_packet(void)
{
    uint8_t buffer[1024];
    int len = recv(mqtt_socket, buffer, sizeof(buffer), 0);
    if (len <= 0) {
        return -1;
    }

    uint8_t type = (buffer[0] >> 4) & 0x0F;

    switch (type) {
        case 0x0D:  // PINGRESP
            printf("收到 PINGRESP\n");
            break;

        case 0x03:  // PUBLISH
            /* 解析主题和载荷 */
            {
                int pos = 1;
                int remaining = buffer[pos++];
                
                int topic_len = (buffer[pos] << 8) | buffer[pos + 1];
                pos += 2;

                char topic[128];
                memcpy(topic, &buffer[pos], topic_len);
                topic[topic_len] = '\0';
                pos += topic_len;

                char payload[1024];
                memcpy(payload, &buffer[pos], len - pos);
                payload[len - pos] = '\0';

                printf("收到消息: topic=%s, payload=%s\n", topic, payload);

                /* 调用回调 */
                if (mqtt_callback) {
                    mqtt_callback(topic, payload);
                }
            }
            break;

        default:
            printf("未知包类型: %d\n", type);
            break;
    }

    return 0;
}

/* 创建 JSON 消息 */
static char* create_json_message(msg_type_t type, const void *data)
{
    cJSON *root = cJSON_CreateObject();

    switch (type) {
        case MSG_TYPE_STATUS: {
            const device_status_t *status = (const device_status_t *)data;
            cJSON_AddStringToObject(root, "type", "status");
            cJSON_AddNumberToObject(root, "temperature", status->temperature);
            cJSON_AddNumberToObject(root, "humidity", status->humidity);
            cJSON_AddNumberToObject(root, "battery", status->battery_level);
            cJSON_AddStringToObject(root, "status", status->status);
            cJSON_AddBoolToObject(root, "alarm", status->alarm_active);
            cJSON_AddNumberToObject(root, "timestamp", time(NULL));
            break;
        }

        case MSG_TYPE_HEARTBEAT:
            cJSON_AddStringToObject(root, "type", "heartbeat");
            cJSON_AddNumberToObject(root, "timestamp", time(NULL));
            break;

        default:
            cJSON_Delete(root);
            return NULL;
    }

    char *json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);

    return json;
}

/* ==================== 网络任务（后台运行） ==================== */
void network_task(void *arg)
{
    printf("网络任务启动\n");

    while (1) {
        /* 检查 WiFi 状态 */
        if (!wifi_config.connected) {
            /* 尝试重连 */
            // TODO: 实现重连逻辑
        }

        /* 检查 MQTT 状态 */
        if (wifi_config.connected && !mqtt_config.connected) {
            /* 尝试连接 MQTT */
            mqtt_connect(mqtt_config.broker, mqtt_config.port,
                        mqtt_config.client_id, mqtt_config.username, mqtt_config.password);
        }

        /* 接收 MQTT 消息 */
        if (mqtt_config.connected) {
            mqtt_parse_packet();

            /* 发送心跳 */
            uint32_t now = time(NULL) * 1000;
            if (now - last_heartbeat_time >= HEARTBEAT_INTERVAL) {
                report_heartbeat();
                last_heartbeat_time = now;
            }
        }

        usleep(100000);  // 100ms
    }
}
