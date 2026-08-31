from pathlib import Path
import json
import cv2
import h5py
import numpy as np
from .core import demos, group, detect_gap
from .sam3_runner import Runner

COLORS = {"peg": (255, 70, 70), "holder": (70, 210, 255), "hand": (90, 230, 120)}

def overlay(image, mask, name, text):
    out = image.copy()
    color = np.asarray(COLORS.get(name, (255, 255, 0)), dtype=np.float32)
    if mask is not None:
        m = mask.astype(bool)
        out[m] = (0.55*out[m] + 0.45*color).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(x) for x in color), 2)
    status = "MATCH" if mask is not None else "NO MATCH"
    cv2.putText(out, f"{name}: {status}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(out, f"{name}: {status}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(out, text[:80], (16, 58), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, text[:80], (16, 58), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1, cv2.LINE_AA)
    return out

def check_prompts(c, max_demos=1):
    out = Path(c["output_dir"]) / "prompt_check"
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(c["sam3"])
    report = []
    with h5py.File(c["dataset_path"], "r") as h:
        selected_demos = demos(h)[:max_demos]
        if not selected_demos:
            raise RuntimeError("No complete demonstrations found")
        for demo in selected_demos:
            for role, serial in (("main", c["main_camera_serial"]), ("secondary", c["secondary_camera_serial"])):
                g = group(h, demo, serial)
                t0, _, _ = detect_gap(g["host_timestamp_ns"][:], c["onset"]["gap_mad_multiplier"], c["onset"]["gap_nominal_multiplier"])
                frame = np.asarray(g["rgb"][t0])
                panels = []
                for name, text in c["text_prompts"][role].items():
                    mask, info = runner.prompt_image(frame, text, out/"work"/demo/role/name)
                    info.update(demo=demo, role=role, frame_index=int(t0), object=name)
                    report.append(info)
                    panels.append(overlay(frame, mask, name, text))
                sheet = np.concatenate(panels, axis=1)
                cv2.imwrite(str(out/f"{demo}_{role}_t0.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    (out/"report.json").write_text(json.dumps(report, indent=2))
    failures = sum(not x["matched"] for x in report)
    print(f"Prompt check: {len(report)-failures}/{len(report)} matched")
    print(f"Review overlays and report in: {out}")
    if failures:
        print(f"Warning: {failures} prompt(s) returned no match")
