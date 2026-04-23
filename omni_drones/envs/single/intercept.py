# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.distributions as D
from typing import Any, cast

import omni_drones.utils.kit as kit_utils

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, Unbounded

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import Iris, MultirotorBase
from omni_drones.utils.torch import (
    euler_to_quaternion,
    normalize,
    quaternion_to_rotation_matrix,
)


class Intercept(IsaacEnv):
    r"""
    A pursuit task where a Hummingbird chases a stationary Iris target.

    ## Observation

    - `pursuer_pos` (3): Hummingbird position in the inertial frame.
    - `pursuer_vel` (6): Hummingbird linear and angular velocity.
    - `pursuer_rot` (9): Hummingbird orientation as a flattened rotation matrix.
    - `relative_direction` (3): Unit vector from pursuer to target in the inertial frame.
    - `relative_velocity` (3): Target linear velocity relative to the pursuer linear velocity.

    ## Reward

    - `closing`: Distance-closing reward, approaching 1 as the drones get closer
      and 0 when they are far apart.
    """

    def __init__(self, cfg, headless):
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reset_thres = cfg.task.get("reset_thres", 15.0)
        self.success_radius = cfg.task.get("success_radius", 0.5)
        self.target_speed_range = cfg.task.get(
            "target_speed_range", [0.8, 1.5])
        self.target_spawn_distance_range = cfg.task.get(
            "target_spawn_distance_range", [4.0, 7.0]
        )
        self.target_bounds = cfg.task.get(
            "target_bounds",
            {
                "min": [-6.0, -6.0, 1.0],
                "max": [6.0, 6.0, 4.0],
            },
        )
        self.time_encoding = cfg.task.get("time_encoding", False)

        super().__init__(cfg, headless)

        self.pursuer.initialize()
        self.target.initialize()

        self.pursuer_init_pos_dist = D.Uniform(
            torch.tensor([-0.5, -0.5, 1.4], device=self.device),
            torch.tensor([0.5, 0.5, 1.9], device=self.device),
        )
        self.pursuer_init_rpy_dist = D.Uniform(
            torch.tensor([-0.15, -0.15, -0.25], device=self.device) * torch.pi,
            torch.tensor([0.15, 0.15, 0.25], device=self.device) * torch.pi,
        )
        self.target_speed_dist = D.Uniform(
            torch.tensor(self.target_speed_range[0], device=self.device),
            torch.tensor(self.target_speed_range[1], device=self.device),
        )
        self.target_spawn_distance_dist = D.Uniform(
            torch.tensor(
                self.target_spawn_distance_range[0], device=self.device),
            torch.tensor(
                self.target_spawn_distance_range[1], device=self.device),
        )

        self.target_bounds_min = torch.tensor(
            self.target_bounds["min"], device=self.device)
        self.target_bounds_max = torch.tensor(
            self.target_bounds["max"], device=self.device)

        self.pursuer_local_pos = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.pursuer_local_rot = torch.zeros(
            self.num_envs, 4, device=self.device)
        self.target_local_pos = torch.zeros(
            self.num_envs, 3, device=self.device)
        self.target_local_rot = torch.zeros(
            self.num_envs, 4, device=self.device)
        self.target_local_vel = torch.zeros(
            self.num_envs, 6, device=self.device)

        self.alpha = 0.8

    def _design_scene(self):
        self.pursuer, self.controller = MultirotorBase.make(
            "Hummingbird", "RateController"
        )
        self.target = Iris()

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        self.pursuer.spawn(translations=[(0.0, 0.0, 1.6)])
        self.target.spawn(translations=[(5.0, 0.0, 2.0)])
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        obs_dim = 3 + 6 + 9 + 3 + 3
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim

        self.observation_spec = Composite({
            "agents": {
                "observation": Unbounded(torch.Size([1, obs_dim]))
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({
            "agents": {
                "action": self.pursuer.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite({
            "agents": {
                "reward": Unbounded(torch.Size([1, 1]))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["pursuer"] = AgentSpec(
            "pursuer",
            1,
            observation_key=cast(Any, ("agents", "observation")),
            action_key=cast(Any, ("agents", "action")),
            reward_key=cast(Any, ("agents", "reward")),
        )

        self.stats_spec = Composite({
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "distance": Unbounded(1),
            "closing_reward": Unbounded(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = self.stats_spec
        self.stats = self.stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        getattr(self.pursuer, "_reset_idx")(env_ids, self.training)

        pursuer_pos = self.pursuer_init_pos_dist.sample(env_ids.shape)
        pursuer_rpy = self.pursuer_init_rpy_dist.sample(env_ids.shape)
        pursuer_rot = euler_to_quaternion(pursuer_rpy)

        spawn_direction = normalize(torch.randn(
            len(env_ids), 3, device=self.device))
        spawn_direction[..., 2] = spawn_direction[..., 2].abs() * 0.35
        spawn_direction = normalize(spawn_direction)
        spawn_distance = self.target_spawn_distance_dist.sample(
            env_ids.shape).unsqueeze(-1)

        target_pos = pursuer_pos + spawn_direction * spawn_distance
        target_yaw = torch.atan2(
            spawn_direction[..., 1], spawn_direction[..., 0])
        target_rot = euler_to_quaternion(torch.stack([
            torch.zeros_like(target_yaw),
            torch.zeros_like(target_yaw),
            target_yaw,
        ], dim=-1))

        self.pursuer_local_pos[env_ids] = pursuer_pos
        self.pursuer_local_rot[env_ids] = pursuer_rot
        self.target_local_pos[env_ids] = target_pos
        self.target_local_rot[env_ids] = target_rot
        self.target_local_vel[env_ids] = 0.0

        self.pursuer.set_world_poses(
            self.pursuer_local_pos[env_ids] + self.envs_positions[env_ids],
            self.pursuer_local_rot[env_ids],
            env_ids,
        )
        self.pursuer.set_velocities(torch.zeros(
            len(env_ids), 6, device=self.device), env_ids)
        self.target.set_world_poses(
            self.target_local_pos[env_ids] + self.envs_positions[env_ids],
            self.target_local_rot[env_ids],
            env_ids,
        )
        self.target.set_velocities(self.target_local_vel[env_ids], env_ids)

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.pursuer.apply_action(actions)

        self.target.set_world_poses(
            self.target_local_pos + self.envs_positions,
            self.target_local_rot,
        )
        self.target.set_velocities(self.target_local_vel)

    def _compute_state_and_obs(self):
        pursuer_pos, pursuer_rot_quat = self.pursuer.get_world_poses(True)
        pursuer_vel = self.pursuer.get_velocities(True)
        target_pos, _ = self.target.get_world_poses(True)
        target_vel = self.target.get_velocities(True)

        pursuer_rot = quaternion_to_rotation_matrix(
            pursuer_rot_quat).flatten(-2)
        pursuer_state = torch.cat(
            [pursuer_pos, pursuer_rot_quat, pursuer_vel], dim=-1)
        target_state = torch.cat([target_pos, self.target.get_world_poses(True)[
                                 1], self.target.get_velocities(True)], dim=-1)

        relative_direction = normalize(target_pos - pursuer_pos)
        relative_velocity = target_vel[..., :3] - pursuer_vel[..., :3]

        obs = [
            pursuer_pos,
            pursuer_vel,
            pursuer_rot,
            relative_direction,
            relative_velocity,
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim))
        obs = torch.cat(obs, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "info": {
                    "drone_state": pursuer_state,
                    "target_state": target_state,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        pursuer_pos, _ = self.pursuer.get_world_poses(True)
        target_pos, _ = self.target.get_world_poses(True)
        pursuer_vel = self.pursuer.get_velocities(True)
        pursuer_state = torch.cat(
            [pursuer_pos, self.pursuer.get_world_poses(True)[1], pursuer_vel], dim=-1)

        # Flatten all positions to [num_envs, 3]
        if pursuer_pos.ndim > 2:
            pursuer_pos = pursuer_pos.reshape(self.num_envs, 3)
        if target_pos.ndim > 2:
            target_pos = target_pos.reshape(self.num_envs, 3)

        distance = torch.norm(target_pos - pursuer_pos, dim=-1, keepdim=True)
        reward = torch.exp(-self.reward_distance_scale * distance)
        reached_target = distance <= self.success_radius

        misbehave = (
            (pursuer_state[..., 2] < 0.15)
            | torch.isnan(pursuer_state).any(-1)
            | (distance > self.reset_thres)
        )
        terminated = misbehave | reached_target
        truncated = (self.progress_buf >=
                     self.max_episode_length - 1).unsqueeze(-1)

        self.stats["distance"].lerp_(distance, 1 - self.alpha)
        self.stats["closing_reward"].lerp_(reward, 1 - self.alpha)
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
