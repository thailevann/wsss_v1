"""
Script to build text prototypes from prompts
"""
import os
import argparse
import torch
import clip
from prototype.text_prototype import build_text_prototypes


def main():
    parser = argparse.ArgumentParser(description="Build text prototypes")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory for data")
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                       help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (default: data_root/text_prototypes_clip.pt)")
    
    args = parser.parse_args()
    
    # Default classes and prompts (from notebook)
    classes = ["Tumor", "Stroma", "Lymphocytic infiltrate", "Necrosis", "Background"]
    
    class2prompts = {
        "Tumor": [
            "irregular glands", "hyperchromatic nuclei", "nuclear crowding", "atypical epithelial cells",
            "pleomorphic tumor epithelium", "high nuclear to cytoplasmic ratio", "invasive epithelial nests",
            "mitotic figures", "abnormal gland formation", "dense tumor clusters"
        ],
        "Stroma": [
            "fibroblast connective tissue", "collagen bundles", "spindle-shaped fibroblasts",
            "eosinophilic fibrous matrix", "myofibroblast stroma", "dense desmoplastic tissue",
            "fibrotic connective region", "reactive fibrous tissue", "stromal hypercellularity", "fibrous connective tissue"
        ],
        "Lymphocytic infiltrate": [
            "small round lymphocytes", "dense lymphoid infiltrate", "monomorphic small cells",
            "immune cell cluster", "band-like lymphocytic infiltration", "basophilic nuclei",
            "peritumoral lymphoid cells", "chronic inflammatory infiltrate", "lymphoid aggregate", "round basophilic nuclei"
        ],
        "Necrosis": [
            "acellular eosinophilic debris", "ghost cells without nuclei", "coagulative necrosis region",
            "granular necrotic material", "cellular remnants", "pale anuclear area",
            "necrotic focus", "amorphous eosinophilic debris", "dead tissue region", "acellular necrotic zone"
        ],
        "Background": [
            "blank slide region without tissue", "white empty background area",
            "unstained glass slide region", "non-tissue area outside the specimen"
        ],
    }
    
    # Load CLIP
    print(f"Loading CLIP model: {args.clip_model}")
    device = args.device if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()
    
    # Build text prototypes (auto-detects feat_dim)
    print("Building text prototypes...")
    text_proto_bank = build_text_prototypes(
        classes=classes,
        class2prompts=class2prompts,
        model=clip_model,
        device=device,
    )
    
    # Save
    output_path = args.output or os.path.join(args.data_root, "text_prototypes_clip.pt")
    os.makedirs(args.data_root, exist_ok=True)
    torch.save(text_proto_bank, output_path)
    
    print(f"\nSaved text prototype bank to: {output_path}")
    print(f"Feature dimension: {text_proto_bank['feat_dim']}")
    for c in classes:
        feat = text_proto_bank["class_text_prototypes"][c]
        if feat.dim() == 1:
            print(f"  {c:24s} prototype dim = {feat.shape[0]}")
        elif feat.dim() == 2:
            print(f"  {c:24s} prototype dim = {feat.shape}")


if __name__ == "__main__":
    main()

