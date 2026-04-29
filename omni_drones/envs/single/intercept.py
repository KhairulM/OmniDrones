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
from torchrl.data import Composite, UnboundedContinuous

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.utils.torch import (
    euler_to_quaternion,
    normalize,
    quaternion_to_rotation_matrix,
)


class Intercept(IsaacEnv):
    r"""
    A pursuit task where a Hummingbird chases a Firefly evader that can hover
    under Lee position control or follow simple scripted trajectories.

    ## Observation

    - `evader_rel_hdg` (3): Relative heading of the evader from the pursuer.
    - `pursuer_lin_vel` (3): Hummingbird linear velocity.
    - `pursuer_rot` (9): Hummingbird orientation as a flattened rotation matrix.
    - `evader_rel_lin_vel` (optional, 3): Evader linear velocity relative to the pursuer linear velocity.
    - `pursuer_pos` (optional, 3): Hummingbird position in world frame.
    - `pursuer_rot_vel` (optional, 3): Hummingbird angular velocity.
    - `time_encoding` (optional, 4): Sinusoidal encoding of the normalized episode time.

    ## Reward

        - `closing`: Distance-closing reward, approaching 1 as the drones get closer
            and 0 when they are far apart.
        - `alignment`: Reward for aligning the pursuer's velocity with the line of
            sight to the target.
        - `approach_speed`: Reward for matching the desired approach speed.
    """

    def __init__(self, cfg, headless):
        """Initialize the interception task, cached configuration, and buffers."""
        self.cfg = cfg

        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reset_thres = cfg.task.get("reset_thres", 15.0)
        self.success_radius = cfg.task.get("success_radius", 0.5)
        self.time_encoding_dim = cfg.task.get("time_encoding_dim", 0)

        pursuer_cfg = cfg.task.pursuer
        evader_cfg = cfg.task.evader

        self.pursuer_model_name = pursuer_cfg.get("model", "Hummingbird")
        self.pursuer_controller_name = pursuer_cfg.get(
            "controller", "RateController")
        self.pursuer_target_speed = pursuer_cfg.get("target_speed", 15.0)
        self.pursuer_use_ab_world_frame = pursuer_cfg.get(
            "use_ab_world_frame", False)
        self.pursuer_use_rot_speed = pursuer_cfg.get("use_rot_speed", False)
        self.evader_speed_range = evader_cfg.get(
            "speed_range",
            [0.8, 1.5],
        )
        self.evader_model_name = evader_cfg.get("model", "Hummingbird")
        self.evader_controller_name = evader_cfg.get(
            "controller", "LeePositionController"
        )
        self.evader_spawn_distance_range = evader_cfg.get(
            "spawn_distance_range",
            [4.0, 7.0],
        )
        self.evader_bounds = evader_cfg.get(
            "bounds",
            {
                "min": [-6.0, -6.0, 1.0],
                "max": [6.0, 6.0, 4.0],
            },
        )
        self.evader_boundary_mode = evader_cfg.get(
            "boundary_mode",
            "bounce",
        )
        self.evader_trajectory_mode = cfg.task.get(
            "evader_trajectory_mode", "hover")
        self.evader_speed = cfg.task.get("evader_speed", 1.0)
        self.evader_use_relative_velocity = evader_cfg.get(
            "use_relative_velocity", False)

        if self.evader_boundary_mode not in {"bounce", "clamp", "wrap"}:
            raise ValueError(
                f"Unsupported evader_boundary_mode: {self.evader_boundary_mode}. "
                "Expected one of: bounce, clamp, wrap."
            )

        super().__init__(cfg, headless)

        self.pursuer.initialize()
        self.evader.initialize()

        self.pursuer_init_pos_dist = D.Uniform(
            torch.tensor([-0.5, -0.5, 1.4], device=self.device),
            torch.tensor([0.5, 0.5, 1.9], device=self.device),
        )
        self.pursuer_init_rpy_dist = D.Uniform(
            torch.tensor([-0.15, -0.15, -0.25], device=self.device) * torch.pi,
            torch.tensor([0.15, 0.15, 0.25], device=self.device) * torch.pi,
        )
        self.evader_speed_dist = D.Uniform(
            torch.tensor(self.evader_speed_range[0], device=self.device),
            torch.tensor(self.evader_speed_range[1], device=self.device),
        )
        self.evader_spawn_distance_dist = D.Uniform(
            torch.tensor(
                self.evader_spawn_distance_range[0], device=self.device),
            torch.tensor(
                self.evader_spawn_distance_range[1], device=self.device),
        )

        self.evader_bounds_min = torch.tensor(
            self.evader_bounds["min"], device=self.device)
        self.evader_bounds_max = torch.tensor(
            self.evader_bounds["max"], device=self.device)

        # Buffers for storing the local states of the pursuer and evader relative to their spawn positions, which are used for computing observations and resetting the drones
        self.pursuer_local_pos = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.pursuer_local_rot = torch.zeros(
            self.num_envs, 1, 4, device=self.device)
        self.pursuer_local_vel = torch.zeros(
            self.num_envs, 1, 6, device=self.device)
        self.evader_local_pos = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.evader_local_rot = torch.zeros(
            self.num_envs, 1, 4, device=self.device)
        self.evader_local_vel = torch.zeros(
            self.num_envs, 1, 6, device=self.device)

        # self.evader_target_pos = torch.zeros(
        #     self.num_envs, 1, 3, device=self.device)
        # self.evader_target_vel = torch.zeros(
        #     self.num_envs, 1, 3, device=self.device)
        # self.evader_target_yaw = torch.zeros(
        #     self.num_envs, 1, 1, device=self.device)

        self.alpha = 0.8

        # Observation components that require storing across steps
        self.evader_rel_hdg = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.evader_rel_lin_vel = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.pursuer_pos = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.pursuer_lin_vel = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.pursuer_rot_vel = torch.zeros(
            self.num_envs, 1, 3, device=self.device)
        self.pursuer_rot = torch.zeros(
            self.num_envs, 1, 9, device=self.device)

    def _design_scene(self):
        """Create the pursuer, evader, and ground plane for each environment."""
        self.pursuer, self.pursuer_controller = MultirotorBase.make(
            self.pursuer_model_name,
            self.pursuer_controller_name,
            device=str(self.device),
            name="pursuer",
        )
        self.evader, self.evader_controller = MultirotorBase.make(
            self.evader_model_name,
            self.evader_controller_name,
            device=str(self.device),
            name="evader",
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        # Spawn drones at default positions; scene cloning handles replication across num_envs
        self.pursuer.spawn(translations=[(0.0, 0.0, 1.6)])
        self.evader.spawn(translations=[(5.0, 0.0, 2.0)])
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        """Define observation, action, reward, and stats specs for the task."""
        pursuer_state_dim = 3 + 9  # lin vel + rot matrix

        if self.pursuer_use_ab_world_frame:
            pursuer_state_dim += 3  # absolute position in world frame
        if self.pursuer_use_rot_speed:
            pursuer_state_dim += 3  # angular velocity

        evader_state_dim = 3  # relative heading

        if self.evader_use_relative_velocity:
            evader_state_dim += 3  # relative linear velocity

        obs_dim = pursuer_state_dim + evader_state_dim

        if self.time_encoding_dim:
            state_dim = obs_dim + self.time_encoding_dim
        else:
            state_dim = obs_dim

        self.observation_spec = Composite({
            "agents": {
                "observation":  UnboundedContinuous(torch.Size([1, obs_dim])),
                "state": UnboundedContinuous(torch.Size([1, state_dim])),
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({
            "agents": {
                # [num_motors = 4] for the pursuer, 0 for the evader since it's controlled by a scripted controller
                "action": self.pursuer.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite({
            "agents": {
                "reward": UnboundedContinuous(torch.Size([1, 1]))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )

        self.stats_spec = Composite({
            "return": UnboundedContinuous(torch.Size([1]), device=self.device),
            "episode_len": UnboundedContinuous(torch.Size([1]), device=self.device),
            "distance": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_closing": UnboundedContinuous(torch.Size([1]), device=self.device),
            "reward_approach": UnboundedContinuous(torch.Size([1]), device=self.device),
            "pursuer_z": UnboundedContinuous(torch.Size([1]), device=self.device),
            "evader_z": UnboundedContinuous(torch.Size([1]), device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.info_spec = Composite({
            "drone_state": UnboundedContinuous(torch.Size([1, 13]), device=self.device),
            # "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
            # "policy_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
            # "prev_prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
        }).expand(self.num_envs).to(self.device)
        # self.info_spec = self.pursuer.info_spec.to(self.device)

        self.observation_spec["stats"] = self.stats_spec
        self.observation_spec["info"] = self.info_spec

        self.stats = self.stats_spec.zero()
        self.info = self.info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset the requested environments and randomize both drones."""
        self.pursuer._reset_idx(env_ids, self.training)
        self.evader._reset_idx(env_ids, self.training)

        pursuer_pos = self.pursuer_init_pos_dist.sample(
            torch.Size([len(env_ids), 1]))
        pursuer_rpy = self.pursuer_init_rpy_dist.sample(
            torch.Size([len(env_ids), 1]))
        pursuer_rot = euler_to_quaternion(pursuer_rpy)

        spawn_direction = normalize(torch.randn(
            len(env_ids), 1, 3, device=self.device))
        spawn_direction[..., 2] = spawn_direction[..., 2].abs() * 0.35
        spawn_direction = normalize(spawn_direction)
        spawn_distance = self.evader_spawn_distance_dist.sample(
            torch.Size([len(env_ids), 1, 1]))

        evader_pos = pursuer_pos + spawn_direction * \
            spawn_distance  # [len(env_ids), 3]
        evader_heading = normalize(torch.randn(
            len(env_ids), 1, 3, device=self.device))
        evader_heading[..., 2] = 0.0
        evader_heading = normalize(evader_heading)
        evader_yaw = torch.atan2(
            evader_heading[..., 1], evader_heading[..., 0])
        evader_rot = euler_to_quaternion(torch.stack([
            torch.zeros_like(evader_yaw),
            torch.zeros_like(evader_yaw),
            evader_yaw,
        ], dim=-1))  # [len(env_ids), 1, 4]

        self.pursuer_local_pos[env_ids] = pursuer_pos
        self.pursuer_local_rot[env_ids] = pursuer_rot
        self.pursuer_local_vel[env_ids] = torch.zeros(
            len(env_ids), 1, 6, device=self.device)
        self.evader_local_pos[env_ids] = evader_pos
        self.evader_local_rot[env_ids] = evader_rot
        self.evader_local_vel[env_ids] = torch.zeros(
            len(env_ids), 1, 6, device=self.device)
        # self.evader_target_pos[env_ids] = evader_pos + \
        #     self.envs_positions[env_ids]
        # self.evader_target_yaw[env_ids] = evader_yaw.unsqueeze(-1)

        self.pursuer.set_world_poses(
            self.envs_positions[env_ids].unsqueeze(
                1) + self.pursuer_local_pos[env_ids],
            self.pursuer_local_rot[env_ids],
            env_ids,
        )
        self.pursuer.set_velocities(self.pursuer_local_vel[env_ids], env_ids)

        self.evader.set_world_poses(
            self.envs_positions[env_ids].unsqueeze(
                1) + self.evader_local_pos[env_ids],
            self.evader_local_rot[env_ids],
            env_ids,
        )
        self.evader.set_velocities(self.evader_local_vel[env_ids], env_ids)

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        """Apply the pursuer policy action and update the evader trajectory before stepping the simulation."""
        actions = self._format_action(tensordict[("agents", "action")])
        self.pursuer_effort = self.pursuer.apply_action(actions)

        # TODO: Replace with proper trajectory tracking for different evader modes instead of simple hover control
        if self.evader_trajectory_mode == "hover":
            evader_state = self.evader.get_state()[..., :13].squeeze(1)
            target_pos = self.evader_local_pos.squeeze(1)
            evader_action = self.evader_controller.compute(
                evader_state, target_pos=target_pos
            )
        else:
            evader_action = torch.zeros(
                self.num_envs, 1, 4, device=self.device)

        self.evader.apply_action(self._format_action(evader_action))

    def _format_action(self, action: torch.Tensor) -> torch.Tensor:
        """Clamp an action tensor and reshape it to the batched controller format."""
        action = torch.nan_to_num(action, 0.0).clamp(-1.0, 1.0)
        return action.reshape(self.num_envs, 1, -1)

    # def _compute_evader_target(self):
    #     """Update the evader target state for the active trajectory mode."""
    #     trajectory_handlers = {
    #         "hover": self._compute_hover_target,
    #         "circular": self._compute_circular_target,
    #         "constant_velocity": self._compute_constant_velocity_target,
    #     }
    #     try:
    #         trajectory_handlers[self.evader_trajectory_mode]()
    #     except KeyError as exc:
    #         raise ValueError(
    #             f"Unknown evader trajectory mode: {self.evader_trajectory_mode}") from exc

    # def _compute_hover_target(self):
    #     """Keep the evader at its current spawn position."""
    #     self.evader_target_pos = self.evader_local_pos + self.envs_positions
    #     self.evader_target_vel = torch.zeros_like(self.evader_target_vel)
    #     self.evader_target_yaw = torch.zeros_like(self.evader_target_yaw)

    # def _compute_circular_target(self):
    #     """Move the evader around a horizontal circular path."""
    #     radius = 2.0
    #     angular_vel = 0.5
    #     t = self.progress_buf.float() * self.cfg.sim.dt
    #     angle = (t * angular_vel).unsqueeze(-1)
    #     center = (self.evader_local_pos + self.envs_positions)[:, :2]

    #     self.evader_target_pos[:, 0] = center[:, 0] + \
    #         radius * torch.cos(angle.squeeze(-1))
    #     self.evader_target_pos[:, 1] = center[:, 1] + \
    #         radius * torch.sin(angle.squeeze(-1))
    #     self.evader_target_pos[:, 2] = self.evader_local_pos[:, 2]

    #     self.evader_target_vel[:, 0] = -radius * \
    #         angular_vel * torch.sin(angle.squeeze(-1))
    #     self.evader_target_vel[:, 1] = radius * \
    #         angular_vel * torch.cos(angle.squeeze(-1))
    #     self.evader_target_vel[:, 2] = 0.0
    #     self.evader_target_yaw = angle

    # def _compute_constant_velocity_target(self):
    #     """Move the evader forward at a fixed speed and clamp it to bounds."""
    #     velocity = torch.tensor(
    #         [self.evader_speed, 0.0, 0.0], device=self.device)
    #     self.evader_target_pos = self.evader_local_pos + \
    #         self.envs_positions + velocity.unsqueeze(0) * self.cfg.sim.dt
    #     self.evader_target_vel = velocity.unsqueeze(
    #         0).expand(self.num_envs, -1)
    #     self.evader_target_yaw = torch.zeros_like(self.evader_target_yaw)
    #     self.evader_target_pos = torch.max(
    #         torch.min(self.evader_target_pos, self.evader_bounds_max),
    #         self.evader_bounds_min,
    #     )

    def _compute_state_and_obs(self):
        """Build the observation tensor and diagnostic state payloads."""
        pursuer_root_state = self.pursuer.get_state(
        )  # [num_envs, 1, state_dim]
        evader_root_state = self.evader.get_state()  # [num_envs, 1, state_dim]

        pursuer_pos = pursuer_root_state[..., :3]
        pursuer_rot_quat = pursuer_root_state[..., 3:7]
        pursuer_vel = pursuer_root_state[..., 7:13]

        evader_pos = evader_root_state[..., :3]
        evader_rot_quat = evader_root_state[..., 3:7]
        evader_vel = evader_root_state[..., 7:13]

        self.evader_rel_hdg = normalize(evader_pos - pursuer_pos)
        # TODO: Change computation to based on how the evader is seen from the pursuer camera frame instead of just the world frame
        self.evader_rel_lin_vel = evader_vel[..., :3] - pursuer_vel[..., :3]
        self.pursuer_pos = pursuer_pos
        self.pursuer_lin_vel = pursuer_vel[..., :3]
        self.pursuer_rot_vel = pursuer_vel[..., 3:6]
        self.pursuer_rot = quaternion_to_rotation_matrix(pursuer_rot_quat).reshape(
            self.num_envs, 1, 9)

        obs = [
            self.evader_rel_hdg,
            self.pursuer_lin_vel,
            self.pursuer_rot,
        ]
        if self.evader_use_relative_velocity:
            obs.append(self.evader_rel_lin_vel)
        if self.pursuer_use_ab_world_frame:
            obs.append(self.pursuer_pos)
        if self.pursuer_use_rot_speed:
            obs.append(self.pursuer_rot_vel)

        state = obs.copy()

        if self.time_encoding_dim:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            state.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))

        obs = torch.cat(obs, dim=-1)
        state = torch.cat(state, dim=-1)

        self.info["drone_state"][:] = pursuer_root_state[..., :13]

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "state": state,
                },
                "stats": self.stats.clone(),
                "info": self.info.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        """Compute reward terms and episode termination flags."""
        pursuer_pos, _ = self.pursuer.get_world_poses(True)
        evader_pos, _ = self.evader.get_world_poses(True)
        pursuer_vel = self.pursuer.get_velocities(True)
        pursuer_rot = self.pursuer.get_world_poses(True)[1]

        evader_velocity = self.evader.get_velocities(True)
        pursuer_pos = self._squeeze_batch(pursuer_pos)
        evader_pos = self._squeeze_batch(evader_pos)
        pursuer_vel = self._squeeze_batch(pursuer_vel)
        pursuer_rot = self._squeeze_batch(pursuer_rot)
        evader_velocity = self._squeeze_batch(evader_velocity)
        pursuer_state = torch.cat(
            [pursuer_pos, pursuer_rot, pursuer_vel], dim=-1)

        distance = torch.norm(evader_pos - pursuer_pos, dim=-1, keepdim=True)
        distance_reward = self._reward_distance_to_evader(
            pursuer_pos, evader_pos)
        alignment_reward = self._reward_align_velocity_to_heading(
            pursuer_vel[..., :3], evader_pos - pursuer_pos
        )
        approach_reward = self._reward_approach_velocity_to_evader(
            pursuer_vel[..., :3], evader_pos - pursuer_pos
        )
        success_interception_reward = self._reward_success_interception(
            pursuer_pos, evader_pos
        )
        interception_time_reward = self._reward_intercept_time(
            pursuer_pos, pursuer_vel, evader_pos, evader_velocity
        )
        # reward = 0.4 * distance_reward + 0.3 * alignment_reward + 0.3 * approach_reward
        # reward = (success_interception_reward + interception_time_reward) / 2.0
        # reward = (approach_reward + distance_reward) / 2.0
        reward = approach_reward
        # reward = (2.0 * reward - 1.0).clamp(-1.0, 1.0)

        reached_target = (distance <= self.success_radius).reshape(
            self.num_envs, 1)
        misbehave = (
            (pursuer_state[..., 2:3] < 0.15)
            | torch.isnan(pursuer_state).any(-1, keepdim=True)
            | (distance > self.reset_thres)
        ).reshape(self.num_envs, 1)

        terminated = misbehave | reached_target
        truncated = (self.progress_buf >= self.max_episode_length - 1)
        truncated = truncated.reshape(self.num_envs, 1)
        done_mask = terminated | truncated

        # terminal_reward = torch.where(
        #     reached_target,
        #     torch.ones_like(reward),
        #     torch.where(misbehave, -torch.ones_like(reward),
        #                 torch.zeros_like(reward)),
        # )

        # reward += terminal_reward

        self.stats["distance"].lerp_(distance, 1 - self.alpha)
        self.stats["reward_closing"].lerp_(distance_reward, 1 - self.alpha)
        self.stats["reward_approach"].lerp_(approach_reward, 1 - self.alpha)
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["pursuer_z"][:] = pursuer_pos[..., 2:3]
        self.stats["evader_z"][:] = evader_pos[..., 2:3]

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": done_mask,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    def _squeeze_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        """Remove a singleton middle dimension from batched simulator tensors."""
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            return tensor.squeeze(1)
        return tensor

    def _reward_distance_to_evader(
        self,
        pursuer_pos: torch.Tensor,
        evader_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Reward the pursuer for reducing the distance to the evader."""
        distance = torch.norm(evader_pos - pursuer_pos, dim=-1, keepdim=True)
        return torch.exp(-self.reward_distance_scale * distance)

    def _reward_intercept_time(
        self,
        pursuer_pos: torch.Tensor,
        pursuer_vel: torch.Tensor,
        evader_pos: torch.Tensor,
        evader_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate time-to-intercept from the current relative motion."""
        relative_pos = evader_pos - pursuer_pos
        relative_vel = evader_vel[..., :3] - pursuer_vel[..., :3]
        relative_speed = torch.norm(relative_vel, dim=-1, keepdim=True) + 1e-6
        time_to_intercept = (relative_pos * normalize(relative_vel)).sum(
            dim=-1, keepdim=True) / relative_speed
        return (-time_to_intercept / self.max_episode_length).clamp(-1.0, 0.0)

    def _reward_success_interception(
        self,
        pursuer_pos: torch.Tensor,
        evader_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Return a binary reward when the pursuer reaches the evader."""
        distance = torch.norm(evader_pos - pursuer_pos, dim=-1, keepdim=True)
        return (distance <= self.success_radius).float()

    def _reward_align_velocity_to_heading(
        self,
        pursuer_velocity: torch.Tensor,
        pursuer_to_evader_heading: torch.Tensor,
    ) -> torch.Tensor:
        """Reward the pursuer for moving in the direction of the evader."""
        pursuer_velocity_direction = normalize(pursuer_velocity)
        evader_heading_direction = normalize(pursuer_to_evader_heading)
        cosine_similarity = (pursuer_velocity_direction * evader_heading_direction).sum(
            dim=-1, keepdim=True
        )
        return ((cosine_similarity + 1.0) / 2.0).clamp(0.0, 1.0)

    def _reward_approach_velocity_to_evader(
        self,
        pursuer_velocity: torch.Tensor,
        pursuer_to_evader_heading: torch.Tensor,
    ) -> torch.Tensor:
        """Reward forward progress toward the evader at the desired speed."""
        approach_speed = (pursuer_velocity * normalize(pursuer_to_evader_heading)).sum(
            dim=-1, keepdim=True
        )
        normalized_speed = approach_speed / self.pursuer_target_speed
        reward = (2.0 * normalized_speed - 1.0).clamp(-1.0, 1.0)
        return reward
