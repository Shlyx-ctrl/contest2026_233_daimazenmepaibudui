/**
 * main.c - 智爱陪伴应用入口
 * 在 openvela 中运行
 */

#include <nuttx/config.h>
#include <stdio.h>
#include <unistd.h>

/* 添加触摸交互头文件 */
#include "robot_ui.h"
#include "touch_ui.h"

/* LVGL 定时器 */
static void lvgl_timer_handler(void)
{
    lv_timer_handler();
}

/* 主函数 */
int main(int argc, char *argv[])
{
    printf("智爱陪伴机器人启动中...\n");

    /* ===== 初始化触摸交互 UI（先初始化） ===== */
    touch_ui_init();

    /* ===== 初始化机器人 UI ===== */
    robot_ui_init();

    /* ===== 添加默认提醒 ===== */
    touch_ui_add_reminder("吃药提醒", "08:00");
    touch_ui_add_reminder("喝水提醒", "10:00");
    touch_ui_add_reminder("散步提醒", "16:00");

    /* ===== 设置初始状态 ===== */
    robot_ui_set_status(ROBOT_STATUS_IDLE);
    robot_ui_set表情(ROBOT表情_HAPPY);
    robot_ui_set_ai_reply("你好！我是智爱陪伴。\n有什么我可以帮你的吗？");

    /* ===== 显示主菜单 ===== */
    touch_ui_show_menu(MENU_TYPE_MAIN);

    printf("智爱陪伴启动完成！\n");

    /* 主循环 */
    while (1) {
        lvgl_timer_handler();
        usleep(5000); // 5ms 刷新周期
    }

    return 0;
}
