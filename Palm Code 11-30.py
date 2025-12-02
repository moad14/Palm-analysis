#!/usr/bin/env python3
"""
run_detection_on_ortho_rgb_excel_shp.py (DEM-based tree height = canopy - ground)

Same as your original script but height_m is computed from DEM as:
    height_m = max_elev_in_box - min_elev_in_box
Fallback: if DEM not available or window empty -> use previous pixel->meter estimate.
"""
import os, math, datetime
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
import cv2
from shapely.geometry import Point
import geopandas as gpd
from pyproj import Transformer, CRS
from ultralytics import YOLO

# ----------------- CONFIG -----------------
MODEL_PATH = r"C:/Users/HP/Desktop/plane/runs/detect/train2/weights/best.pt"
ORTHO_PATH = r"C:/Users/HP/Desktop/plane/data/Orthomosaic_Clip31.tif"
DEM_PATH   = r"C:/Users/HP/Desktop/plane/data/DEM_Clip1/DEM_Clip1.tif"   # تأكد أن هذا المسار صحيح
OUTPUT_DIR = r"C:/Users/HP/Desktop/plane/output_ortho"
EXCEL_PATH = os.path.join(OUTPUT_DIR, "palms_results.xlsx")
SHAPEFILE_PATH = os.path.join(OUTPUT_DIR, "palms_info.shp")
OVERLAY_IMG = os.path.join(OUTPUT_DIR, "overlay_Orthomosaic_Clip31.jpg")

CONF_THRESH = 0.2
IOU_THRESH = 0.45
MAX_DETECT_SIDE = 1600
PIXEL_SIZE_M = None

CROWN_CLASSES = [(0, 3.0, 'Small'), (3.0, 6.0, 'Medium'), (6.0, 999.0, 'Large')]
HEIGHT_CLASSES = [(0, 6.0, 'Low'), (6.0, 12.0, 'Medium'), (12.0, 999.0, 'High')]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- Helpers -----------------
def classify(value, classes):
    for lo, hi, name in classes:
        try:
            if lo <= value < hi:
                return name
        except Exception:
            continue
    return "Unknown"

def ensure_uint8(img):
    if img.dtype == np.uint8:
        return img
    mmin, mmax = float(np.nanmin(img)), float(np.nanmax(img))
    if mmax == mmin:
        return np.clip(img, 0, 255).astype(np.uint8)
    if mmax > 255 or mmin < 0:
        img2 = 255 * (img - mmin) / (mmax - mmin + 1e-9)
        return img2.astype(np.uint8)
    return img.astype(np.uint8)

# ----------------- Main -----------------
def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Opening ortho:", ORTHO_PATH)
    src = rasterio.open(ORTHO_PATH)
    print("Ortho size (w,h):", src.width, src.height, "CRS:", src.crs)

    # Try to open DEM
    dem = None
    transformer_to_dem = None
    try:
        dem = rasterio.open(DEM_PATH)
        print("Opened DEM:", DEM_PATH, "CRS:", dem.crs)
        if src.crs is not None and dem.crs is not None and src.crs != dem.crs:
            transformer_to_dem = Transformer.from_crs(src.crs, dem.crs, always_xy=True)
    except Exception as e:
        print("⚠️ Could not open DEM:", DEM_PATH, " - will fallback to pixel->meter estimate.", e)
        dem = None

    w, h = src.width, src.height
    max_side = max(w, h)
    scale = 1.0
    if max_side > MAX_DETECT_SIDE:
        scale = MAX_DETECT_SIDE / float(max_side)
        new_w = int(math.ceil(w * scale))
        new_h = int(math.ceil(h * scale))
        print(f"Downscaling for detection: scale={scale:.4f}, new size=({new_w},{new_h})")
        img = src.read([1,2,3], out_shape=(3, new_h, new_w), resampling=Resampling.bilinear)
        img = np.dstack((img[0], img[1], img[2])).astype(np.float32)
    else:
        img = src.read([1,2,3])
        img = np.dstack((img[0], img[1], img[2])).astype(np.float32)

    img_rgb = ensure_uint8(img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    print("Running model.predict ...")
    results = model.predict(source=img_bgr, imgsz=MAX_DETECT_SIDE, device='cpu',
                            conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)
    if len(results) == 0:
        print("No results from model.")
        return
    res = results[0]

    master = []
    next_id = 1

    has_crs = src.crs is not None
    if has_crs:
        to_lonlat = Transformer.from_crs(src.crs, CRS.from_epsg(4326), always_xy=True)
        if PIXEL_SIZE_M is None:
            try:
                if src.crs.to_epsg() == 4326:
                    px_deg = abs(src.transform.a)
                    PIXEL = px_deg * 111320
                else:
                    px_size_x = abs(src.transform.a)
                    px_size_y = abs(src.transform.e)
                    PIXEL = (px_size_x + px_size_y)/2.0
                print(f"Pixel size (m): {PIXEL:.6f}")
            except Exception:
                PIXEL = 1.0
        else:
            PIXEL = PIXEL_SIZE_M
    else:
        PIXEL = PIXEL_SIZE_M if PIXEL_SIZE_M else 1.0
        print("No CRS in ortho; using PIXEL_SIZE_M =", PIXEL)

    boxes = getattr(res.boxes, "xyxy", None)
    confs = getattr(res.boxes, "conf", None)
    masks = getattr(res.masks, "data", None) if hasattr(res, "masks") else None

    if boxes is not None:
        boxes = boxes.cpu().numpy()
        confs = confs.cpu().numpy()
    if masks is not None:
        masks = masks.cpu().numpy()

    overlay = img_bgr.copy()
    n_det = 0

    for i_box in range(len(boxes) if boxes is not None else 0):
        x1, y1, x2, y2 = boxes[i_box]
        conf = float(confs[i_box]) if confs is not None else 0.0
        if conf < CONF_THRESH:
            continue

        # scale back to original ortho pixel coords
        x1_o = x1 / scale if scale != 1.0 else x1
        y1_o = y1 / scale if scale != 1.0 else y1
        x2_o = x2 / scale if scale != 1.0 else x2
        y2_o = y2 / scale if scale != 1.0 else y2

        cx_px = (x1_o + x2_o)/2.0
        cy_px = (y1_o + y2_o)/2.0
        height_px = y2_o - y1_o

        crown_px = None
        if masks is not None and masks.shape[0] > i_box:
            mask_res = masks[i_box].astype(np.uint8) * 255
            mask_big = cv2.resize(mask_res, (src.width, src.height), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else mask_res
            try:
                contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours and len(contours) > 0:
                    largest = max(contours, key=cv2.contourArea)
                    (cx_c, cy_c), radius = cv2.minEnclosingCircle(largest)
                    crown_px = float(radius * 2.0)
                    M = cv2.moments(largest)
                    if M.get('m00',0) != 0:
                        cx_px = float(M['m10']/M['m00'])
                        cy_px = float(M['m01']/M['m00'])
            except Exception:
                crown_px = None

        if crown_px is None:
            crown_px = max(x2_o - x1_o, y2_o - y1_o)

        crown_m = crown_px * PIXEL

        # ---------------- DEM: compute tree height = max - min within bbox ----------------
        dem_height_value = None
        if dem is not None and has_crs:
            try:
                # compute bbox in map coords using ortho transform
                left, top = rasterio.transform.xy(src.transform, int(round(y1_o)), int(round(x1_o)), offset='ul')
                right, bottom = rasterio.transform.xy(src.transform, int(round(y2_o)), int(round(x2_o)), offset='lr')

                # transform to DEM CRS if needed
                if transformer_to_dem is not None:
                    left_d, top_d = transformer_to_dem.transform(left, top)
                    right_d, bottom_d = transformer_to_dem.transform(right, bottom)
                else:
                    left_d, top_d = left, top
                    right_d, bottom_d = right, bottom

                # normalize bounds to (left, bottom, right, top)
                minx = min(left_d, right_d)
                maxx = max(left_d, right_d)
                miny = min(bottom_d, top_d)
                maxy = max(bottom_d, top_d)

                # get a window on DEM (boundless to avoid crashes at edges)
                win = from_bounds(minx, miny, maxx, maxy, dem.transform)
                dem_window = dem.read(1, window=win, boundless=True, fill_value=np.nan)

                if dem_window.size == 0:
                    dem_height_value = None
                else:
                    valid = dem_window[~np.isnan(dem_window)]
                    if valid.size == 0:
                        dem_height_value = None
                    else:
                        canopy = float(np.nanmax(valid))
                        ground = float(np.nanmin(valid))
                        # tree height = canopy - ground (meters)
                        dem_height_value = canopy - ground
                        # if negative (bad data) fallback to None
                        if dem_height_value < 0:
                            dem_height_value = None
            except Exception as e:
                dem_height_value = None

        # fallback if dem failed
        if dem_height_value is None:
            height_m = height_px * PIXEL
        else:
            height_m = dem_height_value

        crown_class = classify(crown_m, CROWN_CLASSES)
        height_class = classify(height_m, HEIGHT_CLASSES)

        if has_crs:
            map_x, map_y = rasterio.transform.xy(src.transform, int(round(cy_px)), int(round(cx_px)), offset='center')
            try:
                lon, lat = to_lonlat.transform(map_x, map_y)
            except Exception:
                lon, lat = map_x, map_y
        else:
            map_x, map_y = cx_px * PIXEL, cy_px * PIXEL
            lon, lat = None, None

        palm_id = f"PALM_{next_id:05d}"; next_id += 1
        n_det += 1

        rec = {
            "palm_id": palm_id,
            "x_pixel": round(cx_px,2),
            "y_pixel": round(cy_px,2),
            "crown_diameter_px": round(crown_px,2),
            "crown_diameter_m": round(crown_m,3),
            "crown_class": crown_class,
            "height_px": round(height_px,2),
            "height_m": round(height_m,3),
            "height_class": height_class,
            "detection_confidence": round(conf,3),
            "map_x": map_x, "map_y": map_y,
            "lon": lon, "lat": lat,
            "source_image": os.path.basename(ORTHO_PATH),
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        master.append(rec)

        draw_x1, draw_y1, draw_x2, draw_y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cv2.rectangle(overlay, (draw_x1, draw_y1), (draw_x2, draw_y2), (0,255,0), 2)
        cv2.putText(overlay, palm_id, (draw_x1, max(draw_y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    print(f"Detections: {n_det}")
    overlay_out = OVERLAY_IMG
    if scale != 1.0:
        try:
            ov_up = cv2.resize(overlay, (src.width, src.height), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(overlay_out, ov_up)
        except Exception:
            cv2.imwrite(overlay_out, overlay)
    else:
        cv2.imwrite(overlay_out, overlay)
    print("Saved overlay image:", overlay_out)

    if len(master) == 0:
        print("No detections -> nothing saved.")
        return

    df = pd.DataFrame(master)
    df.to_excel(EXCEL_PATH, index=False, float_format="%.3f")
    print("Excel saved:", EXCEL_PATH)

    try:
        gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df['map_x'], df['map_y'])], crs=src.crs)
        gdf.to_file(SHAPEFILE_PATH, driver='ESRI Shapefile')
        print("Shapefile saved:", SHAPEFILE_PATH)
    except Exception as e:
        print("Could not save shapefile:", e)

    src.close()
    if dem is not None:
        dem.close()
    print("Done.")

if __name__ == "__main__":
    main()
