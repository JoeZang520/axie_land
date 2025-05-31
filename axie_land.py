import cv2
import pyautogui
import numpy as np
import os
import time
import sys
import subprocess


def image(png, threshold=0.8, offset=(0, 0), click_times=1, region=None, color=True, gray_diff_threshold=15):
    if not png.endswith('.png'):
        png += '.png'
    image_path = os.path.join('pic', png)
    if not os.path.exists(image_path):
        print(f"[ERROR] 图片不存在: {image_path}")
        return None

    template = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if template is None:
        print(f"[ERROR] 图片加载失败: {image_path}")
        return None

    region = region or (0, 0, *pyautogui.size())
    x1, y1, x2, y2 = region
    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
    screen_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    if color:
        # 彩色匹配
        result = cv2.matchTemplate(screen_img, template, cv2.TM_CCOEFF_NORMED)
    else:
        # 灰度匹配
        screen_gray = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        # print(f"[MISS] 没有找到 {png}")
        return None

    match_area = screen_img[
        max_loc[1]:max_loc[1] + template.shape[0],
        max_loc[0]:max_loc[0] + template.shape[1]
    ]

    if color:
        diff_rg = np.abs(match_area[:, :, 2] - match_area[:, :, 1])
        diff_rb = np.abs(match_area[:, :, 2] - match_area[:, :, 0])
        diff_gb = np.abs(match_area[:, :, 1] - match_area[:, :, 0])
        mean_diff = np.mean((diff_rg + diff_rb + diff_gb) / 3.0)

        if mean_diff < gray_diff_threshold:
            print(f"[FAIL] {png} 匹配区域颜色太灰（均差≈{mean_diff:.2f}, 未识别出图片")
            return None

    center_x = max_loc[0] + template.shape[1] // 2 + x1 + offset[0]
    center_y = max_loc[1] + template.shape[0] // 2 + y1 + offset[1]

    if click_times > 0:
        for _ in range(click_times):
            pyautogui.click(center_x, center_y)
            time.sleep(1)
        print(f"[ACTION] 点击 {png} {center_x, center_y} {threshold}")

    return (center_x, center_y)


thresholds = {
    "tree1": 0.8,
    "tree2": 0.8,
    "tree3": 0.85,
    "tree4": 0.8,
    "tree5": 0.95,
    "tree6": 0.95,  
    "tree7": 0.9,
    "tree8": 0.9,
    "tree9": 0.9,
    "stone1": 0.9,
    "stone2": 0.9,
    "stone4": 0.9


}
def image_multi(png_list, thresholds=thresholds, region=None, min_x_distance=40, min_y_distance=40, click_times=0, excluded_points=None):
    if isinstance(png_list, str):
        png_list = [png_list]

    if not thresholds:
        raise ValueError("阈值字典 (thresholds) 必须提供")

    region = region or (0, 0, *pyautogui.size())
    x1, y1, x2, y2 = region

    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - x1))
    screen_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

    excluded_points = excluded_points or []
    results = {}

    first_valid_point = None
    first_valid_template = None
    first_valid_threshold = None

    def is_far_enough(cx, cy, points, min_dx, min_dy):
        for px, py, _ in points:
            if abs(cx - px) < min_dx and abs(cy - py) < min_dy:
                return False
        for ex, ey in excluded_points:
            if abs(cx - ex) < min_dx  and abs(cy - ey) < min_dy:
                return False
        return True

    for picture in png_list:
        templates = []
        for file in os.listdir('pic'):  
            if file.startswith(f"{picture}_") and file.endswith('.png'):
                templates.append(os.path.join('pic', file))
          
        if not templates:
            print(f"[ERROR] 未找到任何多模板图片：{picture}_*.png")
            results[picture] = []
            continue

        threshold = thresholds.get(picture)
        if threshold is None:
            print(f"[WARN] 图片 {picture} 没有设置阈值，跳过该角色")
            continue

        all_points = []
        for template_path in templates:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"[ERROR] 图片加载失败: {template_path}")
                continue

            result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= threshold)
            
            # 添加调试信息
            max_val = np.max(result)
            # print(f"[DEBUG] 模板 {template_path} 的最大匹配度: {max_val:.3f}, 阈值: {threshold}")
            
            w, h = template.shape[1], template.shape[0]

            for pt, score in zip(zip(*loc[::-1]), result[loc]):
                cx = pt[0] + w // 2 + x1
                cy = pt[1] + h // 2 + y1

                if is_far_enough(cx, cy, all_points, min_x_distance, min_y_distance):
                    all_points.append((cx, cy, score))
                    # print(f"[DEBUG] 找到匹配点: ({cx}, {cy}), 匹配度: {score:.3f}, 图片: {template_path}")

                    if first_valid_point is None:
                        first_valid_point = (cx, cy)
                        first_valid_template = template_path
                        first_valid_threshold = score

        results[picture] = all_points

    # 点击全局第一个通过筛选的点
    if click_times > 0 and first_valid_point:
        cx, cy = first_valid_point
        print(f"[INFO] 点击匹配点：({first_valid_template} {cx}, {cy})，匹配度：{first_valid_threshold:.3f}")
        for _ in range(click_times):
            pyautogui.click(cx, cy)
            pyautogui.click(cx, cy+25
                            )
            time.sleep(1)
            pyautogui.press('space')

    return results

def wait_for_load(image_names, check_interval: float = 1, threshold=0.8, click_times=1, timeout=45):
    """循环检测任意一张指定图片出现，返回True或False"""
    start_time = time.time()
    print(f"正在加载 {image_names} ... ")

    while True:
        for image_name in image_names:
            pos = image(image_name, threshold=threshold, click_times=click_times, color=True)
            if pos is not None:

                return image_name

        if timeout and (time.time() - start_time) > timeout:
            print(f"加载 {image_names} 超时")
            return None

        time.sleep(check_interval)

def drag(start_pos, end_pos, duration=1):
    start_x, start_y = start_pos
    end_x, end_y = end_pos
    
    # 移动到起始位置
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown(button='left')
    pyautogui.moveTo(end_x, end_y, duration=duration)
    pyautogui.mouseUp(button='left')
    time.sleep(1)

def in_game():
    return image('homeland', offset=(100, 0), gray_diff_threshold=12) is not None

land_path = r"E:\Axie Infinity - Homeland\Homeland.exe"
def enter_game():
    if not in_game():
        print("当前不在游戏中。")
        subprocess.Popen(land_path)
        wait_for_load(["join"])
        wait_for_load(["tab"])
        # wait_for_load(["join"], timeout=30)
        wait_for_load(["acoin"])
        image('M')
        image('x')
   

def close_game():
    subprocess.run(["taskkill", "/f", "/im", "Homeland.exe"], shell=True)
    time.sleep(10)


def collect(tree_count, stone_count):
    """
    同时采集树和石头的函数
    :param tree_count: 采集树的次数
    :param stone_count: 采集石头的次数
    """
    # 按下并保持Shift+Q
    pyautogui.keyDown('shift')
    pyautogui.keyDown('q')
    
    # 采集树
    clicked_points = []
    tree_keys = ['tree1', 'tree2', 'tree3', 'tree4', 'tree5', 'tree6', 'tree7', 'tree8', 'tree9']
    
    for _ in range(tree_count):
        result = image_multi(
            png_list=tree_keys,
            thresholds=thresholds,
            click_times=1,
            excluded_points=clicked_points
        )

        tree_points = []
        for key in tree_keys:
            if result.get(key):
                tree_points = result[key]
                break

        if tree_points:
            cx, cy, _ = tree_points[0]
            clicked_points.append((cx, cy))
        else:
            print("[MISS] 没有可砍的树了。")
            break
    
    print("[INFO] 树木采集结束")
    
    # 采集石头
    clicked_points = []  # 重置已点击的点
    stone_keys = ['stone1', 'stone2', 'stone4']
    
    for _ in range(stone_count):
        result = image_multi(
            png_list=stone_keys,
            thresholds=thresholds,
            click_times=1,
            excluded_points=clicked_points
        )

        stone_points = []
        for key in stone_keys:
            if result.get(key):
                stone_points = result[key]
                break

        if stone_points:
            cx, cy, _ = stone_points[0]
            clicked_points.append((cx, cy))
            pyautogui.click(cx, cy+25)  # 石头需要额外点击
        else:
            print("[MISS] 没有可采的石头了。")
            break
    
    print("[INFO] 石头采集结束")
    
    # 释放按键
    pyautogui.keyUp('q')
    pyautogui.keyUp('shift')

def craft():
    if image('cuddle_kitchen1', click_times=2):
        time.sleep(2)
        image('claim'), time.sleep(1)
        image('ok', color=False), time.sleep(1)
        image('baguette')
        image('craft', click_times=5, color=False)
        pyautogui.press('Esc')
        image('acoin', offset=(-100, 0))
        time.sleep(3)
    else:
        print("未找到cuddle_kitchen1")
    if image('cuddle_kitchen4', click_times=2):
        time.sleep(2)
        if image('#2', click_times=0):
            image('left_arrow'), time.sleep(1)
        image('claim'), time.sleep(1)
        image('ok', color=False), time.sleep(1)
        image('boiled_carrot')
        image('craft', click_times=9, color=False)

        image('right_arrow'), time.sleep(1)
        image('claim'), time.sleep(1)
        image('ok', color=False), time.sleep(1)
        image('boiled_carrot')
        image('craft', click_times=9, color=False)

        pyautogui.press('Esc')
        image('acoin', offset=(-100, 0))
        time.sleep(3)
    else:
        print("未找到cuddle_kitchen4")


def countdown(activity, seconds):
    for i in range(seconds, 0, -1):
        sys.stdout.write(f"\r下一轮{activity}倒计时：{i} 秒")
        sys.stdout.flush()
        time.sleep(1)
    print("\r倒计时结束！      ")

def switch_plot(plot):
    image('plot')
    if plot == '57_119':
        image('acoin', offset=(-420, 280))  # 自己的地
    image(plot)  
    time.sleep(5)
    if plot == '105_128':
        image('acoin', offset=(-340, 280))  # 别人的地
    image(plot)  
    time.sleep(5)
    for _ in range(5):
        pyautogui.scroll(30)
        time.sleep(1)
    pyautogui.press("A")
    pyautogui.hotkey('shift', 'b')
    time.sleep(3)
    


while True:
    enter_game()
    if image('exit', color=False):
        time.sleep(60)
        enter_game()
        

    switch_plot('105_128')
    craft()
    collect(9, 3)  

    switch_plot('57_119')
    craft()
    collect(25, 10)  
    

    countdown("收菜", 1800)













