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
            time.sleep(0.5)
        print(f"[ACTION] 点击 {png}")

    return (center_x, center_y)


thresholds = {
    "tree1": 0.8,
    "tree2": 0.8,
    "tree3": 0.8,
    "tree4": 0.8,
    "stone":0.9

}
def image_multi(png_list, thresholds=thresholds, region=None, min_x_distance=40, min_y_distance=40, click_times=0, excluded_points=None):
    import pyautogui
    import cv2
    import numpy as np
    import os

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
    first_valid_score = None

    def is_far_enough(cx, cy, points, min_dx, min_dy):
        for px, py, _ in points:
            if abs(cx - px) < min_dx and abs(cy - py) < min_dy:
                return False
        for ex, ey in excluded_points:
            if abs(cx - ex) < min_dx and abs(cy - ey) < min_dy:
                return False
        return True

    for picture in png_list:
        templates = []
        i = 1
        while True:
            path = os.path.join('pic', f"{picture}_{i}.png")
            if os.path.exists(path):
                templates.append(path)
                i += 1
            else:
                break

        if not templates:
            print(f"[ERROR] 未找到任何多模板图片：{picture}_1.png, {picture}_2.png 等")
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
            w, h = template.shape[1], template.shape[0]

            for pt, score in zip(zip(*loc[::-1]), result[loc]):
                cx = pt[0] + w // 2 + x1
                cy = pt[1] + h // 2 + y1

                if is_far_enough(cx, cy, all_points, min_x_distance, min_y_distance):
                    all_points.append((cx, cy, score))
                    # print(f"匹配点: ({cx}, {cy}), 匹配度: {score:.3f}, 图片: {template_path}")

                    if first_valid_point is None:
                        first_valid_point = (cx, cy)
                        first_valid_template = template_path
                        first_valid_score = score

        results[picture] = all_points

    # 点击全局第一个通过筛选的点
    if click_times > 0 and first_valid_point:
        cx, cy = first_valid_point
        print(f"[INFO] 点击匹配点：({first_valid_template} {cx}, {cy})，匹配度：{first_valid_score:.3f}")
        for _ in range(click_times):
            pyautogui.click(cx, cy)
            time.sleep(1)
    return results



def in_game():
    return image('homeland', offset=(100, 0), gray_diff_threshold=12) is not None

land_path = r"E:\Axie Infinity - Homeland\Homeland.exe"
def enter_game():
    if not in_game():
        print("当前不在游戏中。")
        subprocess.Popen(land_path)
        time.sleep(10)
def close_game():
    subprocess.run(["taskkill", "/f", "/im", "Homeland.exe"], shell=True)
    time.sleep(10)

enter_game()












