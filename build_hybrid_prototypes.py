"""
Script to build hybrid prototypes from text and vision prototypes
"""
import os
import argparse
import torch
from prototype.hybrid_prototype import build_hybrid_prototypes


def main():
    parser = argparse.ArgumentParser(description="Build hybrid prototypes")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory for data")
    parser.add_argument("--text_proto_path", type=str, default=None,
                       help="Text prototype path (default: data_root/text_prototypes_clip.pt)")
    parser.add_argument("--vision_proto_path", type=str, default=None,
                       help="Vision prototype path (default: data_root/vision_prototypes_learned.pt)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (default: data_root/hybrid_prototypes.pt)")
    parser.add_argument("--feat_dim", type=int, default=None,
                       help="Feature dimension (auto-detected if None)")
    
    args = parser.parse_args()
    
    text_proto_path = args.text_proto_path or os.path.join(args.data_root, "text_prototypes_clip.pt")
    vision_proto_path = args.vision_proto_path or os.path.join(args.data_root, "vision_prototypes_learned.pt")
    
    # Build hybrid prototypes (auto-detects feat_dim if not provided)
    print("Building hybrid prototypes...")
    hybrid_bank = build_hybrid_prototypes(
        text_proto_path=text_proto_path,
        vision_proto_path=vision_proto_path,
        feat_dim=args.feat_dim,
    )
    
    # Save
    output_path = args.output or os.path.join(args.data_root, "hybrid_prototypes.pt")
    os.makedirs(args.data_root, exist_ok=True)
    torch.save(hybrid_bank, output_path)
    
    print(f"\n[OK] Saved hybrid prototypes to: {output_path}")
    print(f"Feature dimension: {hybrid_bank['feat_dim']}")
    for c, proto in hybrid_bank["hybrid_prototypes"].items():
        print(f"  {c:24s}: {tuple(proto.shape)}")


if __name__ == "__main__":
    main()

