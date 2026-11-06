"""Live visualization of a trained policy in the training sim (top-down map + depth the net sees)."""
import argparse
import math
from random import normalvariate

import matplotlib
import torch
from torch.nn import functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='checkpoint0001.pth')
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--speed_mtp', type=float, default=4)
parser.add_argument('--fov_x_half_tan', type=float, default=0.82)
parser.add_argument('--cam_angle', type=int, default=20)
parser.add_argument('--no_odom', default=False, action='store_true')
parser.add_argument('--update_target_continuously', '--update-target-continuously', action='store_true',
                    help='recompute direction to target from the true position each step (no yaw drift)')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--episodes', type=int, default=5, help='episodes to play back to back')
parser.add_argument('--speed', type=float, default=1.0, help='playback speed (1 = real time)')
parser.add_argument('--save', default=None, help='write an mp4/gif instead of opening a window')
args = parser.parse_args()
if args.save:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from env_cuda import Env
from model import Model

if args.seed is not None:
    torch.manual_seed(args.seed)
device = torch.device('cuda')
B = 8
env = Env(B, 64, 48, 0.4, device, fov_x_half_tan=args.fov_x_half_tan, single=True,
          ground_voxels=True, speed_mtp=args.speed_mtp, random_rotation=True,
          cam_angle=args.cam_angle)
model = Model(7 if args.no_odom else 10, 6).to(device).eval()
model.load_state_dict(torch.load(args.resume, map_location=device))


def rollout():
    """Run one episode, return per-step snapshots for drone 0."""
    env.reset()
    model.reset()
    h = None
    act_buffer = [env.act]
    target_v_raw = env.p_target - env.p
    drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
    zeros, ones = torch.zeros_like(drift_av), torch.ones_like(drift_av)
    R_drift = torch.stack([
        torch.cos(drift_av), -torch.sin(drift_av), zeros,
        torch.sin(drift_av), torch.cos(drift_av), zeros,
        zeros, zeros, ones], -1).reshape(B, 3, 3)
    frames = []
    with torch.no_grad():
        for t in range(args.timesteps):
            ctl_dt = normalvariate(1 / 15, 0.1 / 15)
            depth, _ = env.render(ctl_dt)
            clearance = (env.find_vec_to_nearest_pt().norm(2, -1) - env.margin).min(0).values[0].item()
            net_in = F.max_pool2d((3 / depth[:1].clamp(0.3, 24) - 0.6)[:, None], 4, 4)[0, 0]
            frames.append(dict(
                p=env.p[0].cpu().numpy().copy(), v=env.v[0].cpu().numpy().copy(),
                depth=depth[0].cpu().numpy().copy(), clearance=clearance,
                net=net_in.cpu().numpy().copy(),
                fwd=env.R[0, :, 0].cpu().numpy().copy()))
            if args.update_target_continuously:
                target_v_raw = env.p_target - env.p
            else:
                target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
            env.run(act_buffer[t], ctl_dt, target_v_raw)

            fwd = env.R[:, :, 0].clone()
            up = torch.zeros_like(fwd)
            fwd[:, 2] = 0
            up[:, 2] = 1
            fwd = F.normalize(fwd, 2, -1)
            R = torch.stack([fwd, torch.cross(up, fwd), up], -1)
            n = torch.norm(target_v_raw, 2, -1, keepdim=True)
            target_v = target_v_raw / n * torch.minimum(n, env.max_speed)
            state = [torch.squeeze(target_v[:, None] @ R, 1), env.R[:, 2], env.margin[:, None]]
            if not args.no_odom:
                state.insert(0, torch.squeeze(env.v[:, None] @ R, 1))
            state = torch.cat(state, -1)
            x = 3 / depth.clamp(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
            x = F.max_pool2d(x[:, None], 4, 4)
            act, _, h = model(x, state, h)
            a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
            act_buffer.append((a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std)
    world = dict(
        balls=env.balls[0].cpu().numpy(), cyl=env.cyl[0].cpu().numpy(),
        voxels=env.voxels[0].cpu().numpy(), target=env.p_target[0].cpu().numpy(),
        radius=env.drone_radius + env.margin[0].item() * 0)
    return frames, world


fig, (ax, ax_d, ax_n) = plt.subplots(1, 3, figsize=(16, 6.5), gridspec_kw=dict(width_ratios=[2, 1, 1]))


def draw_world(frames, world):
    ax.clear()
    ax_d.clear()
    path = torch.tensor([f['p'] for f in frames]).numpy()
    lo = torch.tensor(path[:, :2]).min(0).values.numpy() - 6
    hi = torch.tensor(path[:, :2]).max(0).values.numpy() + 6
    tx, ty = world['target'][:2]
    lo = [min(lo[0], tx - 2), min(lo[1], ty - 2)]
    hi = [max(hi[0], tx + 2), max(hi[1], ty + 2)]
    for cx, cy, r in world['cyl']:
        if lo[0] - r < cx < hi[0] + r and lo[1] - r < cy < hi[1] + r:
            ax.add_patch(Circle((cx, cy), r, color='forestgreen', alpha=0.6))
    for cx, cy, cz, rx, ry, rz in world['voxels']:
        if rx > 50 or cz - rz > 3.5:
            continue  # roof sentinel / far above flight height
        if lo[0] - rx < cx < hi[0] + rx and lo[1] - ry < cy < hi[1] + ry:
            high = cz + rz > 0.3
            ax.add_patch(Rectangle((cx - rx, cy - ry), 2 * rx, 2 * ry,
                                   color='sienna' if high else 'lightgray', alpha=0.6))
    for cx, cy, cz, r in world['balls']:
        if lo[0] - r < cx < hi[0] + r and lo[1] - r < cy < hi[1] + r and cz + r > 0.3:
            ax.add_patch(Circle((cx, cy), r, color='royalblue', alpha=0.4))
    ax.plot(*world['target'][:2], 'r*', ms=18, label='target')
    ax.plot(path[0, 0], path[0, 1], 'go', label='start')
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    return path


def play():
    for ep in range(args.episodes):
        frames, world = rollout()
        path = draw_world(frames, world)
        trail, = ax.plot([], [], 'k-', lw=1.5)
        drone, = ax.plot([], [], 'ko', ms=9)
        heading = ax.plot([], [], 'r-', lw=2)[0]
        img = ax_d.imshow(frames[0]['depth'], cmap='viridis_r', vmin=0, vmax=15)
        ax_d.set_title('depth seen by drone (m)')
        ax_d.axis('off')
        ax_n.clear()
        img_n = ax_n.imshow(frames[0]['net'], cmap='gray', vmin=-0.6, vmax=2, interpolation='nearest')
        ax_n.set_title('network input 16x12\n(inverse depth, white = near)')
        ax_n.axis('off')
        hit = False
        for k, f in enumerate(frames):
            hit |= f['clearance'] <= 0
            trail.set_data(path[:k + 1, 0], path[:k + 1, 1])
            drone.set_data([f['p'][0]], [f['p'][1]])
            heading.set_data([f['p'][0], f['p'][0] + f['fwd'][0]], [f['p'][1], f['p'][1] + f['fwd'][1]])
            img.set_data(f['depth'])
            img_n.set_data(f['net'])
            speed = float((f['v'] ** 2).sum() ** 0.5)
            status = 'COLLISION' if hit else 'ok'
            ax.set_title(f'episode {ep + 1}  t={k / 15:4.1f}s  speed={speed:4.1f} m/s  '
                         f'alt={f["p"][2]:4.1f} m  clearance={f["clearance"]:4.2f} m  [{status}]',
                         color='red' if hit else 'black')
            if args.save:
                yield
            else:
                plt.pause(1 / 15 / args.speed)


if args.save:
    from matplotlib.animation import FuncAnimation
    gen = play()
    anim = FuncAnimation(fig, lambda _: next(gen, None), frames=args.episodes * args.timesteps, repeat=False)
    anim.save(args.save, fps=15)
    print('saved', args.save)
else:
    for _ in play():
        pass
    plt.show()
