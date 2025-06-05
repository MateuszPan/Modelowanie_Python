import numpy as np
import matplotlib.pyplot as plt
import cv2

N = 3
steps = 1000
dt = 0.01
G = 1.0

initial_conditions = [
    {
        'x': np.array([0.97000436, -0.97000436, 0.0]),
        'y': np.array([-0.24308753, 0.24308753, 0.0]),
        'vx': np.array([0.93240737 / 2, 0.93240737 / 2, -0.93240737]),
        'vy': np.array([0.86473146 / 2, 0.86473146 / 2, -0.86473146])
    }
]

def calculate_average_distance(x, y):
    total_dist = 0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            total_dist += dist
            count += 1
    return total_dist / count

frame_size = (600, 600)
video = cv2.VideoWriter('animacja_do_trajektorii_8.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, frame_size)

def world_to_image(x, y, size):
    scale = 100
    cx, cy = size[0] // 2, size[1] // 2
    return int(cx + x * scale), int(cy - y * scale)

for condition in initial_conditions:
    x = condition['x'].copy()
    y = condition['y'].copy()
    vx = condition['vx'].copy()
    vy = condition['vy'].copy()

    fx = np.zeros(N)
    fy = np.zeros(N)
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            d = np.sqrt(dx**2 + dy**2) + 1e-10
            F = G / d**2
            fx[i] += F * dx / d
            fy[i] += F * dy / d
            fx[j] -= F * dx / d
            fy[j] -= F * dy / d

    x_prev = x - vx * dt + 0.5 * fx * dt**2
    y_prev = y - vy * dt + 0.5 * fy * dt**2

    x_traj = []
    y_traj = []

    for _ in range(steps):
        fx = np.zeros(N)
        fy = np.zeros(N)

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                d = np.sqrt(dx ** 2 + dy ** 2) + 1e-10
                F = G / d ** 2
                fx[i] += F * dx / d
                fy[i] += F * dy / d
                fx[j] -= F * dx / d
                fy[j] -= F * dy / d

        x_new = 2 * x - x_prev + fx * dt ** 2
        y_new = 2 * y - y_prev + fy * dt ** 2

        x_prev = x.copy()
        y_prev = y.copy()
        x = x_new
        y = y_new

        x_traj.append(x.copy())
        y_traj.append(y.copy())

    for step in range(steps):
        frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255
        colors = [(255, 0, 0), (0, 0, 255), (0, 128, 0)]

        for i in range(step):
            for idx in range(N):
                px, py = world_to_image(x_traj[i][idx], y_traj[i][idx], frame_size)
                cv2.circle(frame, (px, py), 2, colors[idx], -1)

        for i in range(N):
            px, py = world_to_image(x_traj[step][i], y_traj[step][i], frame_size)
            cv2.circle(frame, (px, py), 6, colors[i], -1)

        cv2.putText(frame, f'Krok: {step}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        video.write(frame)

video.release()
print("Zapisano animację do pliku 'animacja_do_trajektorii_8.mp4'")
