from pathlib import Path
import shutil
import cv2
import numpy as np


class Runner:
    def __init__(self, c):
        from sam3.model_builder import build_sam3_video_predictor
        kw = {"gpus_to_use": c.get("gpus", [0])}
        if c.get("checkpoint_path"):
            kw["checkpoint_path"] = c["checkpoint_path"]
        if c.get("bpe_path"):
            kw["bpe_path"] = c["bpe_path"]
        self.p = build_sam3_video_predictor(**kw)
        self.c = c

    def np(self, x):
        return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)

    def unpack(self, o):
        ids = self.np(o.get("out_obj_ids", [])).reshape(-1).astype(int)
        m = self.np(o.get("out_binary_masks", []))
        if m.ndim == 4:
            m = m[:, 0]
        if m.ndim == 2:
            m = m[None]
        return ids, m.astype(bool)

    def prompt_image(self, frame, text, wd):
        """Run one text prompt on one image without video propagation."""
        fd = Path(wd)/"frame"
        shutil.rmtree(fd, ignore_errors=True)
        fd.mkdir(parents=True)
        cv2.imwrite(str(fd/"000000.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        sid = self.p.handle_request(dict(
            type="start_session", resource_path=str(fd),
            offload_video_to_cpu=True, offload_state_to_cpu=True))["session_id"]
        try:
            r = self.p.handle_request(dict(
                type="add_prompt", session_id=sid, frame_index=0, text=text,
                output_prob_thresh=self.c.get("output_probability_threshold", .45)))
            ids, masks = self.unpack(r["outputs"])
            if not len(ids):
                return None, {"prompt": text, "matched": False, "candidate_count": 0}
            areas = masks.reshape(len(masks), -1).sum(1)
            selected = int(np.argmax(areas))
            mask = masks[selected]
            return mask, {
                "prompt": text, "matched": True,
                "candidate_count": int(len(ids)),
                "selected_object_id": int(ids[selected]),
                "selected_area_px": int(areas[selected]),
                "selected_area_fraction": float(areas[selected] / mask.size),
            }
        finally:
            self.p.handle_request(dict(type="close_session", session_id=sid))

    def one(self, F, text, t0, wd):
        fd = Path(wd)/"frames"
        shutil.rmtree(fd, ignore_errors=True)
        fd.mkdir(parents=True)
        for i, x in enumerate(F):
            cv2.imwrite(str(fd/f"{i:06d}.jpg"),
                        cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
        sid = self.p.handle_request(dict(
            type="start_session", resource_path=str(fd),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True))["session_id"]
        try:
            r = self.p.handle_request(dict(
                type="add_prompt", session_id=sid, frame_index=t0,
                text=text,
                output_prob_thresh=self.c.get(
                    "output_probability_threshold", .45)))
            ids, masks = self.unpack(r["outputs"])
            if not len(ids):
                raise RuntimeError(f"No instance for prompt: {text}")

            areas = masks.reshape(len(masks), -1).sum(axis=1)
            selected = int(np.argmax(areas))
            oid = int(ids[selected])
            if not np.any(masks[selected]):
                raise RuntimeError(
                    f"Selected instance has an empty prompt mask: {text}")

            out = np.zeros((len(F), *F.shape[1:3]), dtype=bool)
            # Propagation may omit its starting frame. Preserve the valid mask
            # returned directly by add_prompt at t0.
            out[t0] = masks[selected]

            for response in self.p.handle_stream_request(dict(
                    type="propagate_in_video", session_id=sid,
                    propagation_direction="both",
                    start_frame_index=t0)):
                frame_index = int(response["frame_index"])
                prop_ids, prop_masks = self.unpack(response["outputs"])
                matches = np.flatnonzero(prop_ids == oid)
                if len(matches):
                    out[frame_index] = prop_masks[matches[0]]
            return out, oid
        finally:
            self.p.handle_request(dict(type="close_session", session_id=sid))

    def all(self, F, prompts, t0, wd):
        M = {}
        meta = {}
        for n, text in prompts.items():
            M[n], oid = self.one(F, text, t0, Path(wd)/n)
            meta[n] = {"prompt": text, "object_id": oid}
        return M, meta
