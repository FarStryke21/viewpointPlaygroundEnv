import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Callable, Dict, Tuple

# Custom Feature Extractor
class CoverageAgentFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, latent_dim: int = 256):
        super().__init__(observation_space, features_dim=latent_dim)
        
        # Extract dimensions from observation space
        num_faces = observation_space['last_observation'].shape[0]
        coverage_dim = observation_space['coverage_percentage'].shape[0]
        viewpoint_dim = observation_space['last_viewpoint'].shape[0]
        
        # Define subnets for different observation components
        self.last_observation_net = nn.Sequential(
            nn.Linear(num_faces, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim)
        )
        
        self.observation_history_net = nn.Sequential(
            nn.Linear(num_faces, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim)
        )
        
        self.coverage_net = nn.Sequential(
            nn.Linear(coverage_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        self.last_viewpoint_net = nn.Sequential(
            nn.Linear(viewpoint_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Shared network after combining all features
        self.shared_net = nn.Sequential(
            nn.Linear(latent_dim * 2 + 64 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        last_obs_feat = self.last_observation_net(features['last_observation'])
        history_feat = self.observation_history_net(features['observation_history'])
        coverage_feat = self.coverage_net(features['coverage_percentage'])
        viewpoint_feat = self.last_viewpoint_net(features['last_viewpoint'])
        
        # Combine all features
        combined_feat = torch.cat([last_obs_feat, history_feat, coverage_feat, viewpoint_feat], dim=1)
        return self.shared_net(combined_feat)

class CoverageAgentnetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class CoverageAgentPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class = CoverageAgentFeatureExtractor,
            features_extractor_kwargs = dict(latent_dim=256),
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CoverageAgentnetwork(self.features_dim)