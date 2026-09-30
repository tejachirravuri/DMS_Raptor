"""Diagnose why NIQE-switch never triggers — measure actual NIQE variation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from core.niqe import compute_niqe_score

# --- CONFIG ---
VIDEO = None  # will auto-detect from args or prompt
PROXY_SIZES = [64, 128, 160, 320, 640]
MAX_FRAMES = 500
FAST_BETA = 0.30
SLOW_BETA = 0.02

def diagnose(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total // MAX_FRAMES)

    print(f"Video: {video_path}")
    print(f"Total frames: {total}, sampling every {stride} frame(s) -> ~{min(MAX_FRAMES, total)} samples")
    print()

    for ps in PROXY_SIZES:
        scores = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                s = compute_niqe_score(frame, proxy_size=ps)
                scores.append(s)
            idx += 1
            if len(scores) >= MAX_FRAMES:
                break

        arr = np.array(scores)

        # Simulate dual-EMA
        ema_fast = arr[0]
        ema_slow = arr[0]
        max_rise = 0.0
        rise_values = []
        for val in arr:
            ema_fast = FAST_BETA * val + (1 - FAST_BETA) * ema_fast
            ema_slow = SLOW_BETA * val + (1 - SLOW_BETA) * ema_slow
            rise = max(0.0, (ema_fast - ema_slow) / (ema_slow + 1e-9))
            rise_values.append(rise)
            if rise > max_rise:
                max_rise = rise

        rise_arr = np.array(rise_values)

        print(f"proxy_size={ps:>3}:")
        print(f"  NIQE scores: min={arr.min():.2f}, max={arr.max():.2f}, "
              f"mean={arr.mean():.2f}, std={arr.std():.2f}")
        print(f"  Coefficient of variation: {arr.std()/arr.mean()*100:.1f}%")
        print(f"  EMA rise:    min={rise_arr.min():.6f}, max={rise_arr.max():.6f}, "
              f"mean={rise_arr.mean():.6f}")
        print(f"  Max rise observed: {max_rise:.6f}")
        print(f"  Current threshold (niqe_c_high): 0.040000")
        triggered = "YES" if max_rise >= 0.04 else "NO"
        cmp = ">=" if max_rise >= 0.04 else "<"
        print(f"  Would NIQE switch?  {triggered} (max_rise {cmp} 0.04)")

        # Suggest threshold
        p90_rise = np.percentile(rise_arr[rise_arr > 0], 90) if np.any(rise_arr > 0) else 0
        p75_rise = np.percentile(rise_arr[rise_arr > 0], 75) if np.any(rise_arr > 0) else 0
        print(f"  Suggested thresholds: c_high={p90_rise:.4f} (P90 of rise), c_low={p75_rise/3:.4f}")
        print()

    cap.release()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video = sys.argv[1]
    else:
        # Try to find a video in common locations
        for p in [
            r"G:\Teja_Master_Thesis\datasets",
            r"G:\Teja_Master_Thesis\DMS_Raptor",
        ]:
            if os.path.isdir(p):
                for root, dirs, files in os.walk(p):
                    for f in files:
                        if f.endswith(('.mp4', '.avi', '.mkv')):
                            video = os.path.join(root, f)
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
        else:
            print("Usage: python diagnose_niqe.py <video_path>")
            sys.exit(1)

    diagnose(video)
