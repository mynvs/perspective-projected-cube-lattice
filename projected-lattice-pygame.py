import pygame
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from math import gcd
from functools import reduce

pygame.init()
width, height = 500, 500
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("projected-lattice-pygame.py")

DEPTH = 25
INWARD_POSITION = 8
FOV = 97

# generate coords
def generate_coordinates():
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
points = generate_coordinates()

# normalize coordinates to [0, 1]
sums = np.sum(np.abs(points), axis=1)
normalized_sums = (sums - sums.min()) / (sums.max() - sums.min())
normalized_sums = 1 - normalized_sums
GAMMA_VALUE = 2.2
gamma_corrected_sums = normalized_sums**GAMMA_VALUE
# sort points based on sums
sorted_indices = np.argsort(gamma_corrected_sums)
points = points[sorted_indices]
gamma_corrected_sums = gamma_corrected_sums[sorted_indices]

# precompute the projection matrix
aspect_ratio = width / height
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
running = True
clock = pygame.time.Clock()
rotation_quat = R.from_quat([0, 0, 0, 1])
target_rotation_quat = rotation_quat
last_pos = None
point_surface = pygame.Surface((width, height))

# main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            last_pos = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            last_pos = None
        elif event.type == pygame.MOUSEMOTION and last_pos is not None:
            x, y = pygame.mouse.get_pos()
            dx = x - last_pos[0]
            dy = y - last_pos[1]
            rotation_delta = R.from_euler('xy', [dy * 0.01, dx * -0.01], degrees=False)
            target_rotation_quat = rotation_delta * target_rotation_quat
            last_pos = (x, y)

    slerp = Slerp([0, 1], R.from_quat([rotation_quat.as_quat(), target_rotation_quat.as_quat()]))
    rotation_quat = slerp(0.2)

    screen.fill((0, 0, 0))
    point_surface.fill((0, 0, 0))

    rotated_points = rotation_quat.apply(points)
    rotated_points += np.array([0, 0, INWARD_POSITION])
    
    # exclude points behind the camera
    mask = rotated_points[:, 2] > 0
    rotated_points = rotated_points[mask]
    visible_gamma_corrected_sums = gamma_corrected_sums[mask]
    
    projected_points, mask = perspective_projection(rotated_points, projection_matrix)
    projected_points = (projected_points + 1) / 2
    projected_points[:, 0] *= width
    projected_points[:, 1] *= height

    for i, point in enumerate(projected_points):
        x, y = np.rint(point).astype(int)
        if 0 <= x < width and 0 <= y < height:
            grayscale_value = int(visible_gamma_corrected_sums[i] * 255)
            point_surface.set_at((x, y), (grayscale_value, grayscale_value, grayscale_value))

    screen.blit(point_surface, (0, 0))
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
