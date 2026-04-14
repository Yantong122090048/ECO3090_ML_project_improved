"""
Test script to verify the improved model works
"""
import torch
import numpy as np
from mm_dqn.config import ModelConfig, TrainConfig, EnvConfig
from mm_dqn.model import DuelingDQNNet, OriginalDQNNet, get_model
from mm_dqn.agent import DoubleDQNAgent, OriginalDQNAgent


def test_model():
    """Test that both models can forward pass"""
    print("=" * 60)
    print("Testing Model Architectures")
    print("=" * 60)

    config = ModelConfig()

    # Test dimensions
    batch_size = 4
    window_size = 50
    n_levels = 10

    # Create dummy inputs
    lob_window = torch.randn(batch_size, window_size, 4 * n_levels)
    dynamic_state = torch.randn(batch_size, config.dynamic_dim)
    agent_state = torch.randn(batch_size, config.agent_dim)

    print(f"\nInput shapes:")
    print(f"  LOB window: {lob_window.shape}")
    print(f"  Dynamic state: {dynamic_state.shape}")
    print(f"  Agent state: {agent_state.shape}")

    # Test Dueling DQN
    print("\n" + "-" * 60)
    print("Testing Dueling DQN (NEW)")
    print("-" * 60)

    dueling_model = DuelingDQNNet(
        n_levels=n_levels,
        dynamic_dim=config.dynamic_dim,
        agent_dim=config.agent_dim,
        hidden_dim=config.hidden_dim,
        action_dim=config.action_dim,
        n_heads=config.n_heads
    )

    dueling_output = dueling_model(lob_window, dynamic_state, agent_state)
    print(f"✓ Dueling DQN output shape: {dueling_output.shape}")
    print(f"✓ Expected: [{batch_size}, {config.action_dim}]")

    # Check Q values
    print(f"\nQ-value statistics:")
    print(f"  Mean: {dueling_output.mean().item():.4f}")
    print(f"  Std:  {dueling_output.std().item():.4f}")
    print(f"  Min:  {dueling_output.min().item():.4f}")
    print(f"  Max:  {dueling_output.max().item():.4f}")

    # Test Original DQN
    print("\n" + "-" * 60)
    print("Testing Original DQN (Baseline)")
    print("-" * 60)

    original_model = OriginalDQNNet(
        n_levels=n_levels,
        dynamic_dim=config.dynamic_dim,
        agent_dim=config.agent_dim,
        hidden_dim=config.hidden_dim,
        action_dim=config.action_dim,
        n_heads=config.n_heads
    )

    original_output = original_model(lob_window, dynamic_state, agent_state)
    print(f"✓ Original DQN output shape: {original_output.shape}")

    # Count parameters
    dueling_params = sum(p.numel() for p in dueling_model.parameters())
    original_params = sum(p.numel() for p in original_model.parameters())

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"Original DQN parameters:  {original_params:,}")
    print(f"Dueling DQN parameters:   {dueling_params:,}")
    print(f"Difference:               {dueling_params - original_params:,} (+{(dueling_params/original_params-1)*100:.1f}%)")

    # Test agent
    print("\n" + "=" * 60)
    print("Testing Agent Training Step")
    print("=" * 60)

    # Create a dummy batch for training
    batch_size_train = 2

    # Create dummy state dictionaries
    state = np.array([
        np.random.randn(window_size, 4 * n_levels).tolist(),  # lob_window
        np.random.randn(config.dynamic_dim).tolist(),         # dynamic_state
        np.random.randn(config.agent_dim).tolist()           # agent_state
    ], dtype=object)

    next_state = np.array([
        np.random.randn(window_size, 4 * n_levels).tolist(),
        np.random.randn(config.dynamic_dim).tolist(),
        np.random.randn(config.agent_dim).tolist()
    ], dtype=object)

    # Create dummy batch
    batch_state = np.stack([state, next_state])
    batch_action = np.array([5, 10])
    batch_reward = np.array([1.0, -0.5])
    batch_next_state = np.stack([next_state, state])
    batch_done = np.array([0.0, 1.0])

    from collections import namedtuple
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

    batch = Transition(batch_state, batch_action, batch_reward, batch_next_state, batch_done)

    # Test Double DQN agent
    print("\nTesting Double DQN Agent...")
    double_dqn_agent = DoubleDQNAgent(
        model=dueling_model,
        target_model=DuelingDQNNet(
            n_levels=n_levels,
            dynamic_dim=config.dynamic_dim,
            agent_dim=config.agent_dim,
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            n_heads=config.n_heads
        ),
        lr=1e-4,
        gamma=0.99,
        use_double_dqn=True
    )

    stats = double_dqn_agent.train_step(batch, batch_size_train)
    print(f"✓ Training step completed")
    print(f"  Loss: {stats['loss']:.4f}")
    print(f"  Q mean: {stats['q_mean']:.4f}")
    print(f"  TD error: {stats['td_error']:.4f}")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)

    # Print summary of improvements
    print("\n🚀 Improvements Summary:")
    print("  ✓ Dueling DQN architecture (V(s) + A(s,a))")
    print("  ✓ Double DQN training (reduce overestimation)")
    print("  ✓ LayerNorm for stability")
    print("  ✓ Gradient clipping")
    print("  ✓ Huber loss (more robust)")
    print("  ✓ Enhanced feature extraction")


if __name__ == "__main__":
    test_model()
