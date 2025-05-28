from pynput import mouse
import pyperclip


def on_click(x, y, button, pressed):
    if pressed:

        # 获取坐标并复制到剪贴板
        coordinates = f"({x}, {y})"
        pyperclip.copy(coordinates)
        print(f"Coordinates copied to clipboard: {coordinates}")

        # 停止监听
        listener.stop()


# 创建鼠标监听器
with mouse.Listener(on_click=on_click) as listener:
    # 等待鼠标点击事件
    listener.join()
