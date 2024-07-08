import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from imageio.v3 import imwrite
from tqdm import tqdm

WIDTH, HEIGHT = 500, 500
NUM_FRAMES = 101
DEPTH = 75

FOV = 87
INWARD_POSITION = 8
ANTIALIASED_MODE = False


# gif generation settings
FPS = 50
QUALITY = 100
LOSSY_QUALITY = 100
MOTION_QUALITY = 100


GLOBAL_GAMMA_VALUE = 4.5
GAMMA_VALUE = 1.1
def to_linear_rgb(value):
    return value ** GAMMA_VALUE
INV_GAMMA = 1.0 / GAMMA_VALUE
def to_srgb(value):
    return value ** INV_GAMMA
# blend two colors linearly
def blend_colors(color1, color2, t):
    return to_srgb((1 - t) * to_linear_rgb(color1) + t * to_linear_rgb(color2))

# rasterize points with anti-aliasing
def rasterize_points_with_aa(points, lightnesses, width, height, output_filename='points.png'):
    image = np.zeros((height, width, 3))

    def plot_pixel_with_aa(x, y, lightness, image):
        x_floor, y_floor = int(np.floor(x)), int(np.floor(y))
        dx, dy = x - x_floor, y - y_floor
        weights = [(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy]
        for (nx, ny), weight in zip([(0, 0), (1, 0), (0, 1), (1, 1)], weights):
            ix, iy = x_floor + nx, y_floor + ny
            if 0 <= ix < width and 0 <= iy < height:
                image[iy, ix] = blend_colors(image[iy, ix], lightness, weight)

    for (x, y), lightness in zip(points, lightnesses):
        plot_pixel_with_aa(x, y, lightness, image)

    imwrite(output_filename, np.rint(image * 255).astype(np.uint8))

# rasterize points without anti-aliasing
def rasterize_points_no_aa(points, lightnesses, width, height, output_filename='points.png'):
    img = np.zeros((height, width), dtype=np.uint8)

    for i, point in enumerate(points):
        x, y = np.rint(point).astype(int)
        if 0 <= x < width and 0 <= y < height:
            grayscale_value = np.rint(lightnesses[i] * 255).astype(int)
            img[y, x] = grayscale_value

    imwrite(output_filename, img)

# generate coords
def generate_coords():
    n, d, s = np.meshgrid(
        np.arange(-DEPTH, DEPTH + 1),
        np.arange(-DEPTH, DEPTH + 1),
        np.arange(-DEPTH, DEPTH + 1),
        indexing='ij'
    )
    mask = (n != 0) | (d != 0) | (s != 0)
    n, d, s = n[mask], d[mask], s[mask]
    coords = np.stack((n, d, s), axis=-1)
    gcds = np.gcd.reduce(coords, axis=1)
    mask = gcds != 1
    return coords[mask]
points = generate_coords()

# normalize coordinates to [0, 1]
sums = np.sum(np.abs(points), axis=1)
normalized_sums = (sums - sums.min()) / (sums.max() - sums.min())
normalized_sums = 1 - normalized_sums
gamma_corrected_sums = normalized_sums**GLOBAL_GAMMA_VALUE
# sort points based on sums
sorted_indices = np.argsort(gamma_corrected_sums)
points = points[sorted_indices]
gamma_corrected_sums = gamma_corrected_sums[sorted_indices]

# precompute the projection matrix
aspect_ratio = WIDTH / HEIGHT
near = 0.01
far = 100
fov_rad = np.radians(FOV)
f = 1 / np.tan(fov_rad / 2)
fa = f / aspect_ratio
nf = 1 / (near - far)
fb = (far + near) * nf
fc = (2 * far * near) * nf
projection_matrix = np.array([
    [fa, 0, 0, 0],
    [0, f, 0, 0],
    [0, 0, fb, fc],
    [0, 0, -1, 0]
])
# perspective projection
def perspective_projection(points, projection_matrix):
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    projected = points_homogeneous @ projection_matrix.T
    projected /= projected[:, 3][:, np.newaxis]
    mask = (projected[:, 2] > 0)
    return projected[mask][:, :2], mask

# initialize
os.makedirs('outputs', exist_ok=True)
os.system(f'cd outputs && del *.png 2>NUL')
angles = np.array([1, 1, 1], dtype=np.float64)
angles /= np.sqrt(np.sum(angles**2))
angles = angles * 120 / (NUM_FRAMES - 1)
rotation_per_frame = R.from_euler('xyz', angles, degrees=True)
rotation_quat = R.from_quat([0, 0, 0, 1])


# main loop
for frame_count in tqdm(range(NUM_FRAMES)):
    rotation_quat = rotation_quat * rotation_per_frame
    rotated_points = rotation_quat.apply(points)
    rotated_points += np.array([0, 0, INWARD_POSITION])
    
    mask = rotated_points[:, 2] > 0
    rotated_points = rotated_points[mask]
    visible_gamma_corrected_sums = gamma_corrected_sums[mask]
    
    projected_points, mask = perspective_projection(rotated_points, projection_matrix)
    projected_points = (projected_points + 1) / 2
    projected_points[:, 0] *= WIDTH
    projected_points[:, 1] *= HEIGHT

    if ANTIALIASED_MODE:
        rasterize_points_with_aa(projected_points, visible_gamma_corrected_sums, WIDTH, HEIGHT, f'outputs/{frame_count}.png')
    else:
        rasterize_points_no_aa(projected_points, visible_gamma_corrected_sums, WIDTH, HEIGHT, f'outputs/{frame_count}.png')


# create .gif
os.system(f'gifski \
        --quality={QUALITY} \
        --lossy-quality={LOSSY_QUALITY} \
        --motion-quality={MOTION_QUALITY} \
        --extra \
        --fps {FPS} \
        -o output.gif \
        outputs/*.png \
        ')
