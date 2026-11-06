import argparse
import math
from random import normalvariate

import torch
from torch.nn import functional as F

from env_cuda import Env
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--resume', required=True)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--episodes', type=int, default=8, help='number of batches')
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--speed_mtp', type=float, default=4)
parser.add_argument('--fov_x_half_tan', type=float, default=0.82)
parser.add_argument('--cam_angle', type=int, default=20)
parser.add_argument('--no_odom', default=False, action='store_true')
parser.add_argument('--no_yaw_drift', default=False, action='store_true')
parser.add_argument('--update_target_continuously', '--update-target-continuously', action='store_true',
                    help='recompute direction to target from the true position each step (no yaw drift)')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda')
B = args.batch_size
env = Env(B, 64, 48, 0.4, device, fov_x_half_tan=args.fov_x_half_tan, single=True,
          ground_voxels=True, speed_mtp=args.speed_mtp, random_rotation=True,
          cam_angle=args.cam_angle)
model = Model(7 if args.no_odom else 10, 6).to(device).eval()
model.load_state_dict(torch.load(args.resume, map_location=device))

succ, spd, minclr, reached = [], [], [], []
with torch.no_grad():
    for ep in range(args.episodes):
        env.reset()
        model.reset()
        h = None
        act_buffer = [env.act]
        target_v_raw = env.p_target - env.p
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        if args.no_yaw_drift:
            drift_av.zero_()
        zeros, ones = torch.zeros_like(drift_av), torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones], -1).reshape(B, 3, 3)
        dists, speeds = [], []
        for t in range(args.timesteps):
            ctl_dt = normalvariate(1 / 15, 0.1 / 15)
            depth, _ = env.render(ctl_dt)
            dists.append(env.find_vec_to_nearest_pt().norm(2, -1) - env.margin)
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
            speeds.append(env.v.norm(2, -1))
        dist = torch.stack(dists).flatten(0, 1)  # (T * 10 sub-steps), B
        avg = torch.stack(speeds).mean(0)
        ok = (dist > 0).all(0)
        succ.append(ok.float())
        spd.append(avg)
        minclr.append(dist.min(0).values)
        reached.append((env.p - env.p_target).norm(2, -1))
        print(f'batch {ep}: success {ok.float().mean():.3f}  avg_speed {avg.mean():.2f}')

succ, spd, minclr, reached = map(torch.cat, (succ, spd, minclr, reached))
print(f'\n{args.resume}: N={len(succ)}')
print(f'success rate     : {succ.mean():.3f}')
print(f'avg speed (all)  : {spd.mean():.2f} m/s')
print(f'avg speed (succ) : {spd[succ > 0].mean():.2f} m/s')
print(f'min clearance med: {minclr.median():.3f} m')
print(f'final dist to target mean: {reached.mean():.2f} m')
