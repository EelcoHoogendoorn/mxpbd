"""minimal matplotlib rendering of modal bodies"""
import numpy as np
from matplotlib.collections import LineCollection


def draw_bodies(ax, bodies, color='b', alpha=1.0):
	for i, body in enumerate(bodies):
		points = np.asarray(body.world_points())
		edges = np.asarray(body.shape.edges)
		ax.add_collection(LineCollection(points[edges], colors=color, alpha=alpha, linewidths=0.8))


def save_gif(fig, draw_frame, n_frames, path, fps=25, dpi=80, colors=16, supersample=4):
	"""render an animation to a gif quantized to a single shared palette

	Line art on a plain background tolerates a modest palette, and one undithered
	palette shared across all frames compresses far better than the per-frame
	adaptive palettes of the stock pillow writer. Rendering at `supersample` times
	the dpi and downscaling with lanczos gives subpixel-antialiased edges; the
	palette must then be wide enough to carry the antialiasing ramp, or the soft
	edges band and shimmer frame to frame as lines cross pixel boundaries.
	"""
	from PIL import Image
	fig.set_dpi(dpi * supersample)
	images = []
	for i in range(n_frames):
		draw_frame(i)
		fig.canvas.draw()
		im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
		if supersample != 1:
			im = im.resize((im.width // supersample, im.height // supersample), Image.LANCZOS)
		images.append(im)
	# fit the shared palette on a mosaic of sampled frames
	w, h = images[0].size
	sample = images[:: max(1, n_frames // 4)][:4]
	mosaic = Image.new('RGB', (w, h * len(sample)))
	for j, im in enumerate(sample):
		mosaic.paste(im, (0, j * h))
	palette = mosaic.quantize(colors=colors)
	quantized = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
	quantized[0].save(
		path, save_all=True, append_images=quantized[1:],
		duration=round(1000 / fps), loop=0, optimize=True)
