import argparse

parser = argparse.ArgumentParser()

# Physical config
group = parser.add_argument_group("Physical config")
group.add_argument("--ham", type=str, choices=["sk", "rrg"], default="sk", help="Hamiltonian type, default: sk")
group.add_argument("--n", type=int, default=30, help="System size, default: 20")
group.add_argument("--d", type=int, default=3, help="Degree of RRG, default: 3")
group.add_argument("--seed", type=int, default=10, help="Random seed, default: 1")
group.add_argument("--beta-init", type=float, default=0.1, help="Initial beta, default: 0.1")
group.add_argument("--beta-final", type=float, default=5.0, help="Final beta, default: 5.0")
group.add_argument("--beta-interval", type=float, default=0.1, help="Beta interval, default: 0.1")
group.add_argument("--z2", action="store_true", help="Use Z2 symmetry")

# Transformer config
group = parser.add_argument_group("Transformer config")
group.add_argument("--patch-size", type=int, default=1, help="Patch size, default: 1")
group.add_argument("--phy-dim", type=int, default=2, help="Physical dimension, default: 2, must be 2^patch_size")
group.add_argument("--emb-dim", type=int, default=16, help="Embedding dimension, default: 16")
group.add_argument("--mlp-dim", type=int, default=64, help="MLP dimension, default: 64")
group.add_argument("--num-heads", type=int, default=1, help="Number of heads, default: 1")
group.add_argument("--num-layers", type=int, default=1, help="Number of layers, default: 1")
group.add_argument("--use-bias", action="store_true", help="Use bias in transformer")

# NADE config
group = parser.add_argument_group("NADE config")
group.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension, default: 64")

# MADE config
group = parser.add_argument_group("MADE config")
group.add_argument("--num-channels", type=int, default=0, help="Number of channels, default: 0 (FVSBN)")

# Training config
group = parser.add_argument_group("Training config")
group.add_argument(
    "--nn", type=str, default="made", choices=["mf", "nade", "transformer", "made"], help="NN type, default: nade"
)
group.add_argument("--nat-grad", action="store_true", help="Use natural gradient")
group.add_argument("--lr", type=float, default=1e-3, help="Learning rate, default: 1e-3")
group.add_argument("--adaptive-lr", action="store_true", help="Use adaptive learning rate")
group.add_argument("--resample-steps", type=int, default=1, help="Resample steps, default: 1 (No reweighting)")
group.add_argument("--epochs", type=int, default=1000, help="Number of epochs, default: 1000")
group.add_argument("--batch-size", type=int, default=1024, help="Batch size, default: 1024")
group.add_argument("--gpu", type=int, default="0", help="GPU id, default: 0, -1 to disable")
group.add_argument("--path", type=str, default="./out/test/10", help="Output path")
group.add_argument("--lambd", type=float, default=1e-3, help="Damping factor, default: 1e-3")
group.add_argument("--use-tb", action="store_true", help="Use tensorboard")

group.add_argument("--ratio_type", type=str, default="normal", choices=["normal", "softmax", "tanh", "softmaxlog"], help="importance sampling ratio type")
group.add_argument("--clip_type", type=str, default="none", choices=["none", "mean", "meanupper", "meanlower", "dist", "distupper", "distlower"], help="type of clip to use")
group.add_argument("--clip_index", type=float, default=0.2, help="clip boundary")
group.add_argument("--scale_index", type=float, default=1.0, help="scale index to scale the ratio(loss)")
group.add_argument("--clip_penalty", type=str, default="none", choices=["none", "positive", "negative"], help="type of clip penalty to use")
group.add_argument("--forward_KL", action="store_true", help="Use forward KL divergence")
group.add_argument("--ratio_store", action="store_true", help="store the ratios")
group.add_argument("--drop_clip", action="store_true", help="drop ratios to clip")

args = parser.parse_args()
