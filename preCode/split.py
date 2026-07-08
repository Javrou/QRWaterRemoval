"""
真实数据图片切分（二维码矩阵为14*10)
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

warped = None
finished = False
roi_points = []
zoom_scale_corners = 1.0
zoom_scale_warp = 1.0

# 保存四个点
points = []

# 当前显示图片
img = None
display = None

START_INDEX = 1

PAD_RATIO = 0.01

RAW_DIR = "real_data/raw_target"
SAVE_DIR = "../raw_data/real_data/target"
PREVIEW_DIR = "real_data/preview"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)


def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)


def split_qr_images(warped):
    global START_INDEX, finished

    cfg = load_config()
    rows = cfg["rows"]
    cols = cfg["cols"]

    # ROI（warped坐标）
    (x1, y1), (x2, y2) = roi_points
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    roi = warped[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]

    # 网格划分
    row_edges = np.linspace(0, rh, rows + 1).astype(int)
    col_edges = np.linspace(0, rw, cols + 1).astype(int)

    index = START_INDEX
    preview = warped.copy()

    for r in range(rows):
        for c in range(cols):

            sx1, sx2 = col_edges[c], col_edges[c + 1]
            sy1, sy2 = row_edges[r], row_edges[r + 1]

            # padding（按cell比例）
            pad_x = int((sx2 - sx1) * 0.05)
            pad_y = int((sy2 - sy1) * 0.05)

            sx1 = max(0, sx1 - pad_x)
            sy1 = max(0, sy1 - pad_y)
            sx2 = min(rw, sx2 + pad_x)
            sy2 = min(rh, sy2 + pad_y)

            crop = roi[sy1:sy2, sx1:sx2]

            filename = f"{index:04d}.png"
            cv2.imwrite(os.path.join(SAVE_DIR, filename), crop)

            # 映射回 warped 坐标用于 preview
            wx1, wy1 = x1 + sx1, y1 + sy1
            wx2, wy2 = x1 + sx2, y1 + sy2

            cv2.rectangle(preview, (wx1, wy1), (wx2, wy2), (0, 255, 0), 2)
            cv2.putText(
                preview,
                str(index),
                (wx1 + 3, wy1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

            index += 1

    # 保存 preview
    preview_name = os.path.join(
        PREVIEW_DIR,
        f"preview_{START_INDEX:04d}.jpg"
    )
    cv2.imwrite(preview_name, preview)

    saved_num = index - START_INDEX
    START_INDEX = index

    print(f"\n保存完成，共 {saved_num} 张二维码")
    print(f"下一张从 {START_INDEX:04d} 开始")

    finished = True

    # 安全关闭窗口
    try:
        cv2.destroyWindow("Warp")
    except:
        pass


def mouse_callback(event, x, y, flags, param):
    global points, display, zoom_scale_corners

    if event == cv2.EVENT_MOUSEWHEEL:
        # 滚轮缩放，每次 10%（兼容新旧 OpenCV）
        try:
            raw_delta = cv2.getMouseWheelDelta(flags)
            delta = 0.1 if raw_delta > 0 else -0.1
        except AttributeError:
            delta = 0.1 if flags > 0 else -0.1
        zoom_scale_corners = max(0.2, min(5.0, zoom_scale_corners + delta))
        refresh_corners_display()
        return

    if event == cv2.EVENT_LBUTTONDOWN:

        # 最多选四个点
        if len(points) >= 4:
            return

        # 转换到原图坐标存储
        orig_x = int(x / zoom_scale_corners)
        orig_y = int(y / zoom_scale_corners)
        points.append((orig_x, orig_y))

        # 画红点
        cv2.circle(display, (x, y), 8, (0, 0, 255), -1)

        # 标号
        cv2.putText(
            display,
            str(len(points)),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.imshow("Select 4 Corners", display)

        print(f"Point {len(points)} : ({orig_x}, {orig_y})")

        if len(points) == 4:

            print("\n四个点已选择完成：")

            for i, p in enumerate(points):
                print(f"{i + 1}: {p}")

            global warped

            warped = warp_paper(img, points)

            cv2.namedWindow("Warp", cv2.WINDOW_NORMAL)
            wh, ww = warped.shape[:2]
            warp_win_h = min(wh, 900)
            warp_win_w = int(ww * warp_win_h / wh)
            cv2.resizeWindow("Warp", warp_win_w, warp_win_h)
            cv2.imshow("Warp", warped)

            if not os.path.exists("config.json"):
                save_config()

            print("\n请选择二维码区域：")
            print("① 左上")
            print("② 右下")

            cv2.setMouseCallback(
                "Warp",
                roi_callback
            )


def roi_callback(event, x, y, flags, param):
    global roi_points, warped, zoom_scale_warp

    if event == cv2.EVENT_MOUSEWHEEL:
        # 滚轮缩放，每次 10%（兼容新旧 OpenCV）
        try:
            raw_delta = cv2.getMouseWheelDelta(flags)
            delta = 0.1 if raw_delta > 0 else -0.1
        except AttributeError:
            delta = 0.1 if flags > 0 else -0.1
        zoom_scale_warp = max(0.2, min(5.0, zoom_scale_warp + delta))
        refresh_warp_display()
        return

    if event == cv2.EVENT_LBUTTONDOWN:

        # 转换到原图坐标存储
        orig_x = int(x / zoom_scale_warp)
        orig_y = int(y / zoom_scale_warp)
        roi_points.append((orig_x, orig_y))

        print(f"ROI Point {len(roi_points)} : ({orig_x}, {orig_y})")

        if len(roi_points) == 2:
            (x1, y1), (x2, y2) = roi_points

            preview = warped.copy()

            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.imshow("Warp", preview)

            if not os.path.exists("config.json"):
                save_config()

            split_qr_images(warped)

            print("Done.")


def reset():
    global points, display, zoom_scale_corners

    points = []
    zoom_scale_corners = 1.0
    display = img.copy()

    cv2.imshow("Select 4 Corners", display)

    print("已重置，请重新点击四个角。")



def refresh_corners_display():
    """根据 zoom_scale_corners 重新生成 Select 4 Corners 窗口的显示"""
    global display, zoom_scale_corners, img, points
    if img is None:
        return
    h, w = img.shape[:2]
    nw = max(100, int(w * zoom_scale_corners))
    nh = max(100, int(h * zoom_scale_corners))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    display = resized.copy()
    for idx, (px, py) in enumerate(points):
        sx = int(px * zoom_scale_corners)
        sy = int(py * zoom_scale_corners)
        cv2.circle(display, (sx, sy), 8, (0, 0, 255), -1)
        cv2.putText(display, str(idx + 1), (sx + 10, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("Select 4 Corners", display)


def refresh_warp_display():
    """根据 zoom_scale_warp 重新生成 Warp 窗口的显示"""
    global warped, zoom_scale_warp, roi_points
    if warped is None:
        return
    h, w = warped.shape[:2]
    nw = max(100, int(w * zoom_scale_warp))
    nh = max(100, int(h * zoom_scale_warp))
    resized = cv2.resize(warped, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw = resized.copy()
    for idx, (px, py) in enumerate(roi_points):
        sx = int(px * zoom_scale_warp)
        sy = int(py * zoom_scale_warp)
        cv2.circle(dw, (sx, sy), 6, (0, 255, 0), -1)
    if len(roi_points) == 2:
        (x1, y1), (x2, y2) = roi_points
        sx1 = int(x1 * zoom_scale_warp)
        sy1 = int(y1 * zoom_scale_warp)
        sx2 = int(x2 * zoom_scale_warp)
        sy2 = int(y2 * zoom_scale_warp)
        cv2.rectangle(dw, (sx1, sy1), (sx2, sy2), (0, 255, 0), 3)
    cv2.imshow("Warp", dw)

def warp_paper(image, pts):
    """
    pts顺序：
    左上
    右上
    右下
    左下
    """

    pts = np.array(pts, dtype=np.float32)

    (tl, tr, br, bl) = pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)

    warped = cv2.warpPerspective(
        image,
        M,
        (width, height),
        flags=cv2.INTER_CUBIC
    )

    return warped


def save_config():
    config = {
        "rows": 14,
        "cols": 10,
        "margin_ratio": 0.01
    }

    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("配置已保存")


# ===========================
# 主程序
# ===========================
image_list = sorted(
    Path(RAW_DIR).glob("*.jpg"),
    key=lambda x: int(x.stem)
)

print(f"发现 {len(image_list)} 张图片")

for img_path in image_list:
    finished = False

    print("\n==============================")
    print("Current:", img_path.name)
    print("==============================")

    points.clear()
    roi_points.clear()

    img = cv2.imread(str(img_path))

    display = img.copy()

    # 按图片比例设置初始窗口
    ih, iw = img.shape[:2]
    win_h = min(ih, 900)
    win_w = int(iw * win_h / ih)

    cv2.namedWindow("Select 4 Corners", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select 4 Corners", win_w, win_h)

    cv2.imshow("Select 4 Corners", display)

    cv2.setMouseCallback(
        "Select 4 Corners",
        mouse_callback
    )

    while True:

        key = cv2.waitKey(20) & 0xff

        if key == 27:
            exit()

        if finished:
            break

        # ---- 键盘缩放 ----
        if key in (ord('+'), ord('=')):
            if warped is not None:
                zoom_scale_warp = min(5.0, zoom_scale_warp + 0.1)
                refresh_warp_display()
            else:
                zoom_scale_corners = min(5.0, zoom_scale_corners + 0.1)
                refresh_corners_display()

        if key == ord('-'):
            if warped is not None:
                zoom_scale_warp = max(0.2, zoom_scale_warp - 0.1)
                refresh_warp_display()
            else:
                zoom_scale_corners = max(0.2, zoom_scale_corners - 0.1)
                refresh_corners_display()

        if key == ord('0'):
            if warped is not None:
                zoom_scale_warp = 1.0
                refresh_warp_display()
            else:
                zoom_scale_corners = 1.0
                refresh_corners_display()

cv2.destroyAllWindows()
raise SystemExit
