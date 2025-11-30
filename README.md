## Cấu trúc thư mục

```
midl/
├── model/              # Model definitions
│   ├── classnet.py     # ClassNet++ model
│   └── clip_encoder.py # CLIP encoder utilities
├── prototype/          # Prototype building modules
│   ├── text_prototype.py
│   ├── visual_prototype.py
│   ├── hybrid_prototype.py
│   └── vision_learner.py
├── utils.py           # Utility functions
├── build_text_prototypes.py      # Script 1: Build text prototypes
├── build_visual_prototypes.py    # Script 2: Build visual prototypes
├── train_vision_prototypes.py   # Script 3: Train vision prototypes
├── build_hybrid_prototypes.py   # Script 4: Build hybrid prototypes
├── train.py            # Script 5: Train ClassNet++
└── eval.py             # Script 6: Evaluate model
```

## Cách sử dụng

### Bước 1: Build Text Prototypes

```bash
python build_text_prototypes.py --data_root /path/to/data
```

Tùy chọn:
- `--clip_model`: CLIP model name (default: "ViT-B/16")
- `--device`: Device to use (default: "cuda")
- `--output`: Output path (default: data_root/text_prototypes_clip.pt)

**Tự động phát hiện**: Feature dimension sẽ được tự động phát hiện từ CLIP model.

### Bước 2: Build Visual Prototypes

```bash
python build_visual_prototypes.py --data_root /path/to/data
```

Tùy chọn:
- `--train_dir`: Training directory (default: data_root/training)
- `--n_prototypes`: Number of prototypes per class (default: 16)
- `--batch_size`: Batch size (default: 64)
- `--feat_dim`: Feature dimension (auto-detected if None)

**Tự động phát hiện**: Feature dimension sẽ được tự động phát hiện từ CLIP model hoặc từ text prototypes nếu đã có.

### Bước 3: Train Vision Prototypes

```bash
python train_vision_prototypes.py --data_root /path/to/data
```

Tùy chọn:
- `--text_proto_path`: Path to text prototypes (auto-detected)
- `--visual_proto_path`: Path to visual prototypes (auto-detected)
- `--num_epochs_stage1`: Epochs for stage 1 (default: 100)
- `--num_epochs_stage2`: Epochs for stage 2 (default: 100)
- `--lr_stage1`: Learning rate stage 1 (default: 0.02)
- `--lr_stage2`: Learning rate stage 2 (default: 0.01)

**Tự động phát hiện**: Feature dimension sẽ được tự động phát hiện từ loaded prototypes.

### Bước 4: Build Hybrid Prototypes

```bash
python build_hybrid_prototypes.py --data_root /path/to/data
```

Tùy chọn:
- `--text_proto_path`: Path to text prototypes (auto-detected)
- `--vision_proto_path`: Path to learned vision prototypes (auto-detected)
- `--feat_dim`: Feature dimension (auto-detected if None)

**Tự động phát hiện**: Feature dimension sẽ được tự động phát hiện từ loaded prototypes.

### Bước 5: Train ClassNet++

```bash
python train.py --data_root /path/to/data
```

Tùy chọn:
- `--train_dir`: Training directory (default: data_root/training)
- `--hybrid_proto_path`: Path to hybrid prototypes (auto-detected)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of epochs (default: 5)
- `--lr`: Learning rate (default: 1e-4)
- `--val_split`: Validation split ratio (default: 0.2)

**Tự động phát hiện**: 
- Input dimension (INPUT_DIM) tự động phát hiện từ CLIP model
- Prototype dimension (PROTO_DIM) tự động phát hiện từ hybrid prototypes

### Bước 6: Evaluate

```bash
python eval.py --data_root /path/to/data
```

Tùy chọn:
- `--val_img_dir`: Validation image directory (default: data_root/test/img)
- `--val_mask_dir`: Validation mask directory (default: data_root/test/mask)
- `--hybrid_proto_path`: Path to hybrid prototypes (auto-detected)
- `--checkpoint_path`: Path to model checkpoint (auto-detected)
- `--scales`: Multi-scale evaluation scales (default: [1.0, 0.8, 1.2])
- `--caa_iters`: CAA iteration count (default: 5)
- `--caa_attn_power`: CAA attention power (default: 3)
- `--caa_thr`: CAA threshold (default: 0.20)
- `--conf_thr`: Confidence threshold (default: 0.05)




```bash
# 1. Build text prototypes
python build_text_prototypes.py --data_root ./BCSS-WSSS

# 2. Build visual prototypes
python build_visual_prototypes.py --data_root ./BCSS-WSSS

# 3. Train vision prototypes
python train_vision_prototypes.py --data_root ./BCSS-WSSS

# 4. Build hybrid prototypes
python build_hybrid_prototypes.py --data_root ./BCSS-WSSS

# 5. Train ClassNet++
python train.py --data_root ./BCSS-WSSS --num_epochs 5

python train.py \
  --data_root /path/to/data \
  --viz_image_paths /path/to/img1.png /path/to/img2.png \
  --viz_output_dir /path/to/output_dir \
  [các tham số train khác...]

# 6. Evaluate
python eval.py --data_root ./BCSS-WSSS
```

## Xử lý các chiều khác nhau

Hệ thống tự động xử lý các trường hợp sau:

1. **Text prototype shape**:
   - `[512]` → Tự động xử lý
   - `[2, 512]` → Tự động xử lý (giữ nguyên shape)
   - `[10, 512]` → Tự động xử lý (per-prompt features)

2. **Visual prototype shape**:
   - `[16, 512]` → Tự động xử lý
   - `[512]` → Tự động xử lý (thêm dimension nếu cần)

3. **Hybrid prototype shape**:
   - Tự động broadcast và combine các prototype với chiều khác nhau
   - Output luôn là `[K_v, K_t, D]` với D được tự động phát hiện

## Lưu ý

- Tất cả các script đều tự động phát hiện feature dimension từ CLIP model hoặc từ các file đã lưu
- Nếu bạn thay đổi chiều của text prototype (ví dụ từ `[512]` sang `[2, 512]`), các bước sau sẽ tự động adapt
- Các file checkpoint và prototype đều lưu `feat_dim` để đảm bảo consistency

## Yêu cầu

- Python 3.7+
- PyTorch
- CLIP (OpenAI)
- numpy
- PIL
- tqdm
- matplotlib
- opencv-python
- scikit-learn

