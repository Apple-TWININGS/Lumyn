"""ScientificExporter：把粒子轨迹导出为科研标准格式。"""
from __future__ import annotations
import os
import json
import csv
import numpy as np


class ScientificExporter:
    """导出 N 体轨迹：CSV（必测）+ FITS（可选，需 astropy）。"""

    @staticmethod
    def to_csv(trajectories: dict, output_path: str, metadata: dict = None,
               time_step: float = 0.01):
        """trajectories: {'pos':(T,N,3), 'vel':(T,N,3), 'mass':(N,)}"""
        pos = np.asarray(trajectories["pos"], dtype=np.float64)
        vel = np.asarray(trajectories.get("vel", np.zeros_like(pos)), dtype=np.float64)
        mass = np.asarray(trajectories["mass"], dtype=np.float64)
        T, N, _ = pos.shape

        with open(output_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "particle_id", "x", "y", "z", "vx", "vy", "vz", "mass"])
            for t in range(T):
                for i in range(N):
                    w.writerow([round(t * time_step, 6), i,
                                round(pos[t, i, 0], 6), round(pos[t, i, 1], 6), round(pos[t, i, 2], 6),
                                round(vel[t, i, 0], 6), round(vel[t, i, 1], 6), round(vel[t, i, 2], 6),
                                round(float(mass[i]), 6)])
        if metadata:
            meta_path = output_path.replace(".csv", "_meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
        return output_path

    @staticmethod
    def to_fits(trajectories: dict, output_path: str, metadata: dict = None):
        """FITS 导出（天文标准，可选依赖）。"""
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError("FITS 需要安装 astropy: pip install astropy")
        pos = np.asarray(trajectories["pos"], dtype=np.float64)
        mass = np.asarray(trajectories["mass"], dtype=np.float64)
        hdu = fits.PrimaryHDU(pos)
        hdu.header["NMASSES"] = len(mass)
        if metadata:
            for k, v in metadata.items():
                key = k.upper()[:8]
                hdu.header[key] = v if isinstance(v, (int, float, str)) else str(v)
        mass_hdu = fits.ImageHDU(mass, name="MASS")
        hdul = fits.HDUList([hdu, mass_hdu])
        hdul.writeto(output_path, overwrite=True)
        return output_path
