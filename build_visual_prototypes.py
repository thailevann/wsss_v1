"""
Script to build visual prototypes from training images
"""
import os
import argparse
import torch
import clip
from prototype.visual_prototype import build_visual_prototypes


def main():
    parser = argparse.ArgumentParser(description="Build visual prototypes")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory for data")
    parser.add_argument("--train_dir", type=str, default=None,
                       help="Training directory (default: data_root/training)")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                       help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--n_prototypes", type=int, default=16,
                       help="Number of prototypes per class")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for feature extraction")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (default: data_root/visual_prototypes_clip.pt)")
    parser.add_argument("--feat_dim", type=int, default=None,
                       help="Feature dimension (auto-detected if None)")
    parser.add_argument("--text_proto_path", type=str, default=None,
                       help="Path to text prototypes for CAM guidance (default: data_root/text_prototypes_clip.pt)")
    parser.add_argument("--sim_threshold", type=float, default=0.2,
                       help="Initial cosine similarity threshold vs. text prototype")
    parser.add_argument("--refine_threshold", type=float, default=0.35,
                       help="Refinement cosine threshold vs. coarse visual prototype")
    parser.add_argument("--cam_topk", type=float, default=0.3,
                       help="Fraction of top CAM tokens to pool for regional features")
    
    args = parser.parse_args()
    
    classes = ["Tumor", "Stroma", "Lymphocytic infiltrate", "Necrosis", "Background"]
    train_dir = args.train_dir or os.path.join(args.data_root, "training")
    
    # Load CLIP
    print(f"Loading CLIP model: {args.clip_model}")
    device = args.device if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()
    clip_model.float()  # ensure float32 weights to match float32 inputs
    
    # Build visual prototypes (auto-detects feat_dim if not provided)
    print("Building visual prototypes...")
    visual_proto_bank = build_visual_prototypes(
        train_dir=train_dir,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        classes=classes,
        device=device,
        n_prototypes_per_class=args.n_prototypes,
        batch_size=args.batch_size,
        feat_dim=args.feat_dim,
        text_proto_path=args.text_proto_path or os.path.join(args.data_root, "text_prototypes_clip.pt"),
        sim_threshold=args.sim_threshold,
        refine_threshold=args.refine_threshold,
        cam_topk=args.cam_topk,
    )
    
    # Save
    output_path = args.output or os.path.join(args.data_root, "visual_prototypes_clip.pt")
    os.makedirs(args.data_root, exist_ok=True)
    torch.save(visual_proto_bank, output_path)
    
    print(f"\nSaved visual prototype bank to: {output_path}")
    print(f"Feature dimension: {visual_proto_bank['metadata']['feat_dim']}")
    for c, proto in visual_proto_bank["visual_prototypes"].items():
        print(f"  {c:24s}: {tuple(proto.shape)}")


if __name__ == "__main__":
    main()

