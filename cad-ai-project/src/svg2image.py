import subprocess

def svg_to_png_cli(svg_path: str, png_path: str, scale: float = 4.0):
    zoom_factor = int(scale * 96)  # resvg --dpi scales from 96 base
    cmd = [
        "resvg",
        "--dpi", str(zoom_factor),   # or use --width 2000 for exact pixels
        # "--background", "white",
        str(svg_path),
        str(png_path)
    ]
    subprocess.run(cmd, check=True)
    print(f"Converted with resvg CLI: {png_path}")

if __name__ == '__main__':
    # Usage
    svg_to_png_cli("storage/images/floor_4_5.svg", "output.png", scale=4.0)
