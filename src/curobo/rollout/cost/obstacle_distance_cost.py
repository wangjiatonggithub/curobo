#!/usr/bin/env python
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
from dataclasses import dataclass
from typing import Optional, Union

# Third Party
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollision
from curobo.rollout.cost.cost_base import CostBase, CostConfig


@dataclass
class ObstacleDistanceCostConfig(CostConfig):
    """Configuration for terminal obstacle distance cost."""

    #: WorldCollision instance to use for distance queries.
    world_coll_checker: Optional[WorldCollision] = None

    #: Penalty distance (meters). Cost is active when distance < threshold_distance.
    threshold_distance: Union[torch.Tensor, float] = 0.05

    #: Activation distance for ESDF query. Typically 0.0 for true distance.
    activation_distance: Union[torch.Tensor, float] = 0.0

    #: Sum distance across obstacles in ESDF query.
    sum_collisions: bool = True

    #: Compute ESDF signed distance (positive inside, negative outside).
    compute_esdf: bool = True

    def __post_init__(self):
        if isinstance(self.threshold_distance, float):
            self.threshold_distance = self.tensor_args.to_device([self.threshold_distance])
        if isinstance(self.activation_distance, float):
            self.activation_distance = self.tensor_args.to_device([self.activation_distance])
        return super().__post_init__()


class ObstacleDistanceCost(CostBase, ObstacleDistanceCostConfig):
    """Terminal cost based on distance to nearest obstacle."""

    def __init__(self, config: ObstacleDistanceCostConfig):
        ObstacleDistanceCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self._collision_query_buffer = CollisionQueryBuffer()
        self._distance_weight = torch.ones_like(self.weight)

    def forward(self, robot_spheres, env_query_idx: Optional[torch.Tensor] = None):
        if self.world_coll_checker is None:
            return torch.zeros(
                (robot_spheres.shape[0], robot_spheres.shape[1]),
                device=robot_spheres.device,
                dtype=robot_spheres.dtype,
            )
        self._collision_query_buffer.update_buffer_shape(
            robot_spheres.shape, self.tensor_args, self.world_coll_checker.collision_types
        )
        dist = self.world_coll_checker.get_sphere_distance(
            robot_spheres,
            self._collision_query_buffer,
            self._distance_weight,
            self.activation_distance,
            env_query_idx=env_query_idx,
            return_loss=False,
            sum_collisions=self.sum_collisions,
            compute_esdf=self.compute_esdf,
        )
        # ESDF returns positive inside obstacles, negative outside.
        distance_outside = -dist
        min_distance = torch.min(distance_outside, dim=-1).values
        penalty = torch.relu(self.threshold_distance - min_distance)

        cost = torch.zeros_like(min_distance)
        cost[:, -1] = penalty[:, -1]
        cost = self.weight * cost
        return cost