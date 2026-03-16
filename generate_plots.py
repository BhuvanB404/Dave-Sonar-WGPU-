#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sonar frame plots and non-zero diagnostics from SonarRawData CSV files."
    )
    parser.add_argument(
        "--base",
        default="/home/bb/sonar-gpu/artifacts/sonar_raw",
        help="Directory containing SonarRawData_*.csv and SonarRawData_beam_angles.csv",
    )
    parser.add_argument(
        "--prefix",
        default="sonar",
        help="Filename prefix for generated PNG and report artifacts",
    )
    return parser.parse_args()


def load_complex_csv(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            vals = [complex(v.replace("i", "j")) for v in line.split(",")]
            data.append(vals)
    return np.array(data, dtype=complex)


def load_beam_angles(beam_file, n_beams):
    beam_angles = np.genfromtxt(beam_file, delimiter=",", comments="#")
    beam_angles = np.asarray(beam_angles).flatten()
    return beam_angles[:n_beams]


def load_run_max_distance(base, fallback):
    run_conditions = os.path.join(base, "run_conditions.json")
    if not os.path.exists(run_conditions):
        return fallback
    try:
        with open(run_conditions, "r", encoding="utf-8") as f:
            data = json.load(f)
        value = (
            data.get("sensor_params", {}).get("maxDistance_m")
            or data.get("sensor_params", {}).get("range_max_m")
        )
        if value is None:
            return fallback
        return float(value)
    except Exception:
        return fallback


def render_polar_scatter(csv_path, beam_file, out_png, title):
    data = load_complex_csv(csv_path)
    n_beams = data.shape[1] - 1
    ranges = data[:, 0].real
    echo = data[:, 1 : n_beams + 1]
    mag = np.abs(echo) * np.sqrt(3.0)
    db = 20.0 * np.log10(mag + 1e-10)
    beam_angles = load_beam_angles(beam_file, n_beams)

    rg, bg = np.meshgrid(ranges, beam_angles, indexing="ij")
    x = rg * np.cos(bg)
    y = rg * np.sin(bg)

    plt.figure(figsize=(9, 6))
    vmax = np.nanmax(db)
    vmin = vmax - 60.0
    plt.scatter(x.flatten(), y.flatten(), c=db.flatten(), s=3, cmap="hot", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Echo Level [dB]")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(title)
    plt.gca().set_facecolor("black")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def render_dave_style_frame(csv_path, beam_file, out_png, max_distance):
    data = load_complex_csv(csv_path)
    n_beams = data.shape[1] - 1
    ranges = data[:, 0].real
    echo = data[:, 1 : n_beams + 1]
    beam_angles = load_beam_angles(beam_file, n_beams)

    intensity_image = np.zeros((echo.shape[0], echo.shape[1]), dtype=np.uint8)
    range_res = float(ranges[1] - ranges[0]) if len(ranges) > 1 else max_distance
    n_effective_ranges = int(np.ceil(max_distance / max(range_res, 1e-9)))
    radius = intensity_image.shape[0]
    origin = (intensity_image.shape[1] // 2, int(intensity_image.shape[0]))
    bin_thickness = int(2 * np.ceil(radius / max(n_effective_ranges, 1)))

    angles = []
    for beam in range(n_beams):
        center = float(beam_angles[beam])
        if beam == 0:
            end = float((beam_angles[beam + 1] + center) / 2.0) if n_beams > 1 else center
            begin = 2.0 * center - end
        elif beam == n_beams - 1:
            begin = angles[beam - 1][2]
            end = 2.0 * center - begin
        else:
            begin = angles[beam - 1][2]
            end = float((beam_angles[beam + 1] + center) / 2.0)
        angles.append((begin, center, end))

    theta_shift = 1.5 * np.pi
    max_power = float(np.max(np.abs(echo)))
    max_db = 20.0 * np.log10(max_power + 1e-10)
    min_db = max_db - 60.0
    db_span = max(max_db - min_db, 1e-12)

    for range_index, range_value in enumerate(ranges):
        if range_value > max_distance:
            continue
        rad = int(float(radius) * float(range_value) / max_distance)
        for beam in range(n_beams):
            power_db = 20.0 * np.log10(np.abs(echo[range_index, n_beams - 1 - beam]) + 1e-10)
            intensity = int(255.0 * (power_db - min_db) / db_span)
            intensity = max(0, min(255, intensity))
            begin = (angles[beam][0] + theta_shift) * 180.0 / np.pi
            end = (angles[beam][2] + theta_shift) * 180.0 / np.pi
            cv2.ellipse(
                intensity_image,
                origin,
                (rad, rad),
                0.0,
                begin,
                end,
                int(intensity),
                bin_thickness,
            )

    image_color = cv2.applyColorMap(intensity_image, cv2.COLORMAP_HOT)
    cv2.imwrite(out_png, image_color)


def render_beam_range_heatmap(csv_path, out_png):
    data = load_complex_csv(csv_path)
    echo = data[:, 1:]
    db = 20.0 * np.log10(np.abs(echo) + 1e-10)
    vmax = np.nanmax(db)
    vmin = vmax - 60.0

    plt.figure(figsize=(10, 5))
    plt.imshow(db, aspect="auto", origin="lower", cmap="hot", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Echo Level [dB]")
    plt.xlabel("Beam index")
    plt.ylabel("Range bin")
    plt.title("Beam-Range Heatmap (Frame 1)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def render_range_profile_peaks(valid_files, out_png):
    peaks = []
    idxs = []
    for f in valid_files:
        data = load_complex_csv(f)
        echo = np.abs(data[:, 1:])
        peak_profile = np.max(echo, axis=1)
        peaks.append(peak_profile)
        m = re.search(r"_(\d+)\.csv$", os.path.basename(f))
        idxs.append(int(m.group(1)) if m else len(idxs) + 1)

    stacked = np.array(peaks)
    plt.figure(figsize=(10, 5))
    plt.plot(np.max(stacked, axis=0), linewidth=1.2)
    plt.xlabel("Range bin")
    plt.ylabel("Peak amplitude across frames")
    plt.title("Range Profile Peaks")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def main():
    args = parse_args()
    base = os.path.abspath(args.base)
    beam_file = os.path.join(base, "SonarRawData_beam_angles.csv")
    files = sorted(glob.glob(os.path.join(base, "SonarRawData_*.csv")))

    if not files:
        raise SystemExit(f"No SonarRawData_*.csv found in {base}")
    if not os.path.exists(beam_file):
        raise SystemExit(f"Missing beam angles file: {beam_file}")

    valid_files = []
    per_frame = []
    for f in files:
        try:
            d = load_complex_csv(f)
            if d.ndim == 2 and d.shape[0] > 0 and d.shape[1] > 2:
                valid_files.append(f)
                mag = np.abs(d[:, 1:])
                zero_count = int(np.count_nonzero(mag == 0.0))
                total_count = int(mag.size)
                per_frame.append(
                    {
                        "file": os.path.basename(f),
                        "shape": [int(d.shape[0]), int(d.shape[1])],
                        "max_mag": float(np.max(mag)),
                        "mean_mag": float(np.mean(mag)),
                        "zero_values": zero_count,
                        "zero_fraction": float(zero_count / max(total_count, 1)),
                        "all_zero": bool(np.all(mag == 0.0)),
                    }
                )
        except Exception:
            pass

    if not valid_files:
        raise SystemExit(f"No valid sonar frame CSV files found in {base}")

    first = valid_files[0]
    mid = valid_files[len(valid_files) // 2]
    last = valid_files[-1]
    first_data = load_complex_csv(first)
    max_distance = load_run_max_distance(base, float(first_data[-1, 0].real))

    render_dave_style_frame(first, beam_file, os.path.join(base, f"{args.prefix}_frame_first.png"), max_distance)
    render_dave_style_frame(mid, beam_file, os.path.join(base, f"{args.prefix}_frame_mid.png"), max_distance)
    render_dave_style_frame(last, beam_file, os.path.join(base, f"{args.prefix}_frame_last.png"), max_distance)

    render_polar_scatter(
        first,
        beam_file,
        os.path.join(base, f"{args.prefix}_polar_first.png"),
        f"Polar Echo Scatter: {os.path.basename(first)}",
    )
    render_polar_scatter(
        mid,
        beam_file,
        os.path.join(base, f"{args.prefix}_polar_mid.png"),
        f"Polar Echo Scatter: {os.path.basename(mid)}",
    )
    render_polar_scatter(
        last,
        beam_file,
        os.path.join(base, f"{args.prefix}_polar_last.png"),
        f"Polar Echo Scatter: {os.path.basename(last)}",
    )

    render_beam_range_heatmap(
        first,
        os.path.join(base, f"{args.prefix}_frame1_beam_range_heatmap.png"),
    )
    render_range_profile_peaks(
        valid_files,
        os.path.join(base, f"{args.prefix}_range_profile_peaks.png"),
    )

    peaks = []
    means = []
    idxs = []
    for f in valid_files:
        d = load_complex_csv(f)
        e = np.abs(d[::8, 1:])
        peaks.append(float(np.max(e)))
        means.append(float(np.mean(e)))
        m = re.search(r"_(\d+)\.csv$", os.path.basename(f))
        idxs.append(int(m.group(1)) if m else len(idxs) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(idxs, peaks, label="Peak echo (linear)", linewidth=1.3)
    plt.plot(idxs, means, label="Mean echo (linear)", linewidth=1.3)
    plt.xlabel("Frame index")
    plt.ylabel("Amplitude")
    plt.title("Sonar Echo Trend Across Captured Frames")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base, f"{args.prefix}_frame_trend.png"), dpi=170)
    plt.close()

    zero_frames = [f for f in per_frame if f["all_zero"]]
    summary = {
        "base": base,
        "frames_found": len(files),
        "frames_valid": len(valid_files),
        "all_zero_frames": len(zero_frames),
        "any_all_zero_frame": bool(zero_frames),
        "max_zero_fraction_any_frame": float(max(f["zero_fraction"] for f in per_frame)),
        "min_mean_mag_any_frame": float(min(f["mean_mag"] for f in per_frame)),
        "per_frame": per_frame,
        "generated_plots": [
            f"{args.prefix}_frame_first.png",
            f"{args.prefix}_frame_mid.png",
            f"{args.prefix}_frame_last.png",
            f"{args.prefix}_frame_trend.png",
            f"{args.prefix}_polar_first.png",
            f"{args.prefix}_polar_mid.png",
            f"{args.prefix}_polar_last.png",
            f"{args.prefix}_frame1_beam_range_heatmap.png",
            f"{args.prefix}_range_profile_peaks.png",
        ],
    }

    summary_path = os.path.join(base, f"{args.prefix}_quality_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = os.path.join(base, f"{args.prefix}_quality_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Sonar Quality Summary ({args.prefix})\n\n")
        f.write(f"- Base path: `{base}`\n")
        f.write(f"- Frames found: {summary['frames_found']}\n")
        f.write(f"- Frames valid: {summary['frames_valid']}\n")
        f.write(f"- Any all-zero frame: {summary['any_all_zero_frame']}\n")
        f.write(f"- All-zero frame count: {summary['all_zero_frames']}\n")
        f.write(f"- Max zero fraction in any frame: {summary['max_zero_fraction_any_frame']:.6f}\n")
        f.write(f"- Min mean magnitude in any frame: {summary['min_mean_mag_any_frame']:.6f}\n")
        f.write("\n## Generated Plots\n")
        for p in summary["generated_plots"]:
            f.write(f"- `{p}`\n")

    print(f"Generated plots and quality report in {base}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary MD:   {report_path}")


if __name__ == "__main__":
    main()
