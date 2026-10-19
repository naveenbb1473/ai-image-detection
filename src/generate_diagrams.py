import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch
import numpy as np

def create_use_case_diagram(output_paths):
    fig, ax = plt.subplots(figsize=(19, 13), dpi=300)
    ax.set_xlim(0, 190)
    ax.set_ylim(0, 130)
    ax.axis('off')
    
    # Background
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    
    # Title Header
    ax.text(95, 124, "AI-Generated Image Detection & XAI — Use Case Diagram", 
            fontsize=20, weight='bold', ha='center', va='center', color='#0F172A', fontfamily='sans-serif')
    ax.text(95, 119.5, "System Boundary, Primary/Secondary Actors & Core Functional Modules", 
            fontsize=12, style='italic', ha='center', va='center', color='#475569', fontfamily='sans-serif')

    # System Boundary Box
    sys_box = FancyBboxPatch((40, 6), 110, 108, boxstyle="round,pad=1.5,rounding_size=2.0",
                             facecolor='#FFFFFF', edgecolor='#3B82F6', linewidth=2.5, linestyle='-', zorder=1)
    ax.add_patch(sys_box)
    
    # System Boundary Label
    sys_badge = FancyBboxPatch((43, 107), 48, 5.5, boxstyle="round,pad=0.5,rounding_size=0.8",
                               facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=1.2, zorder=2)
    ax.add_patch(sys_badge)
    ax.text(67, 109.7, "System: AI-Image-Detection Pipeline", fontsize=11, weight='bold',
            ha='center', va='center', color='#1D4ED8', fontfamily='sans-serif', zorder=3)

    # Actor drawing helper
    def draw_actor(x, y, name, role="Actor", color="#1E293B", badge_color="#E2E8F0"):
        # Head
        head = Circle((x, y + 5), 2.2, facecolor=color, edgecolor=color, zorder=4)
        ax.add_patch(head)
        # Body
        ax.plot([x, x], [y + 2.8, y - 3.5], color=color, lw=2.5, zorder=4)
        # Arms
        ax.plot([x - 3.5, x + 3.5], [y + 0.5, y + 0.5], color=color, lw=2.5, zorder=4)
        # Legs
        ax.plot([x, x - 3], [y - 3.5, y - 8.5], color=color, lw=2.5, zorder=4)
        ax.plot([x, x + 3], [y - 3.5, y - 8.5], color=color, lw=2.5, zorder=4)
        
        # Name Tag Badge
        name_box = FancyBboxPatch((x - 13, y - 16.5), 26, 6.2, boxstyle="round,pad=0.4,rounding_size=1.0",
                                  facecolor=badge_color, edgecolor=color, linewidth=1.5, zorder=4)
        ax.add_patch(name_box)
        ax.text(x, y - 13.0, name, fontsize=10, weight='bold', ha='center', va='center', color=color, fontfamily='sans-serif', zorder=5)
        ax.text(x, y - 15.5, f"«{role}»", fontsize=8, style='italic', ha='center', va='center', color='#64748B', fontfamily='sans-serif', zorder=5)

    # Primary Actor 1: End User / Researcher (Left Top)
    draw_actor(18, 92, "User / Evaluator", role="Primary Actor", color="#0F172A", badge_color="#F1F5F9")
    
    # Primary Actor 2: ML Engineer / Researcher (Left Bottom)
    draw_actor(18, 38, "ML Engineer", role="Primary Actor", color="#0F172A", badge_color="#F1F5F9")
    
    # Secondary Actor 3: PyTorch Hub / DINOv2 (Right Top)
    draw_actor(172, 85, "PyTorch Hub", role="External System", color="#4338CA", badge_color="#EEF2FF")

    # Secondary Actor 4: Streamlit Web UI (Right Bottom)
    draw_actor(172, 35, "Streamlit Dashboard", role="Web Interface", color="#047857", badge_color="#ECFDF5")

    # Use case bubbles helper
    use_cases = {}
    def draw_use_case(uc_id, cx, cy, w, h, title, category="inference"):
        colors = {
            "inference": {"bg": "#EFF6FF", "edge": "#2563EB", "txt": "#1E40AF"},
            "xai": {"bg": "#FAF5FF", "edge": "#9333EA", "txt": "#6B21A8"},
            "data": {"bg": "#FFFBEB", "edge": "#D97706", "txt": "#92400E"},
            "train": {"bg": "#F0FDF4", "edge": "#16A34A", "txt": "#166534"},
        }
        cfg = colors.get(category, colors["inference"])
        
        # Ellipse-like rounded box
        uc_patch = FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.8,rounding_size=3.0",
                                  facecolor=cfg["bg"], edgecolor=cfg["edge"], linewidth=1.8, zorder=3)
        ax.add_patch(uc_patch)
        ax.text(cx, cy, title, fontsize=9.5, weight='bold', ha='center', va='center', color=cfg["txt"],
                fontfamily='sans-serif', multialignment='center', zorder=4)
        use_cases[uc_id] = (cx, cy, w, h)

    # Use Cases within Boundary:
    # User / Dashboard group
    draw_use_case("UC1", 66, 96, 26, 7.5, "Upload Image\n(JPEG / PNG / WebP)", "inference")
    draw_use_case("UC2", 102, 96, 28, 7.5, "Detect AI vs Real Image\n(Single Image Inference)", "inference")
    draw_use_case("UC3", 136, 85, 24, 7.5, "Extract DINOv2\nViT Embeddings (384-d)", "inference")
    draw_use_case("UC4", 102, 80, 26, 7.5, "Classify Features\n(Linear / MLP Head)", "inference")
    
    # XAI group
    draw_use_case("UC5", 70, 72, 28, 7.5, "Generate XAI Visual\nAttention Heatmap", "xai")
    draw_use_case("UC6", 112, 66, 27, 7.5, "Compute CLS Attention\n(16x16 Patch Weights)", "xai")
    draw_use_case("UC7", 70, 56, 28, 7.5, "Execute Perturbation\nFaithfulness Analysis", "xai")

    # Data Quality & Splitting group
    draw_use_case("UC8", 66, 38, 26, 7.5, "Detect Exact & Near\nDuplicates (dHash/LSH)", "data")
    draw_use_case("UC9", 102, 42, 27, 7.5, "Generate Disjoint Split\nManifests (DSU)", "data")
    draw_use_case("UC10", 138, 52, 24, 7.5, "Pre-extract & Cache\nDataset Tensors (.pt)", "data")

    # Training & Evaluation group
    draw_use_case("UC11", 66, 20, 26, 7.5, "Train Classification Head\n(Linear / MLP + Val F1)", "train")
    draw_use_case("UC12", 102, 18, 27, 7.5, "Evaluate Model on Test Set\n(ROC-AUC, F1, Precision)", "train")
    draw_use_case("UC13", 138, 22, 24, 7.5, "Generate Metrics &\nConfusion Matrix Plot", "train")

    # Connecting Lines Helper
    def connect_actor_uc(actor_pos, uc_id, color="#334155"):
        ax.plot([actor_pos[0], use_cases[uc_id][0] - use_cases[uc_id][2]/2], 
                [actor_pos[1], use_cases[uc_id][1]], color=color, lw=1.5, zorder=2)

    def connect_actor_uc_right(actor_pos, uc_id, color="#4338CA"):
        ax.plot([actor_pos[0], use_cases[uc_id][0] + use_cases[uc_id][2]/2], 
                [actor_pos[1], use_cases[uc_id][1]], color=color, lw=1.5, zorder=2)

    # Actor Connections
    # User / Evaluator
    connect_actor_uc((18 + 13, 92 - 13), "UC1", "#0F172A")
    connect_actor_uc((18 + 13, 92 - 13), "UC2", "#0F172A")
    connect_actor_uc((18 + 13, 92 - 13), "UC5", "#0F172A")
    
    # ML Engineer
    connect_actor_uc((18 + 13, 38 - 13), "UC7", "#0F172A")
    connect_actor_uc((18 + 13, 38 - 13), "UC8", "#0F172A")
    connect_actor_uc((18 + 13, 38 - 13), "UC9", "#0F172A")
    connect_actor_uc((18 + 13, 38 - 13), "UC10", "#0F172A")
    connect_actor_uc((18 + 13, 38 - 13), "UC11", "#0F172A")
    connect_actor_uc((18 + 13, 38 - 13), "UC12", "#0F172A")

    # PyTorch Hub
    connect_actor_uc_right((172 - 13, 85 - 13), "UC3", "#4338CA")
    connect_actor_uc_right((172 - 13, 85 - 13), "UC6", "#4338CA")
    
    # Streamlit Web UI
    connect_actor_uc_right((172 - 13, 35 - 13), "UC1", "#047857")
    connect_actor_uc_right((172 - 13, 35 - 13), "UC2", "#047857")
    connect_actor_uc_right((172 - 13, 35 - 13), "UC5", "#047857")

    # UML Relationships helper (include / extend)
    def draw_relation(src_id, dst_id, rel_type="«include»", color="#64748B", rad=0.0):
        src_pt = (use_cases[src_id][0], use_cases[src_id][1])
        dst_pt = (use_cases[dst_id][0], use_cases[dst_id][1])
        
        arrow = FancyArrowPatch(src_pt, dst_pt,
                                connectionstyle=f"arc3,rad={rad}",
                                linestyle="--", linewidth=1.4, color=color,
                                arrowstyle="-|>", mutation_scale=14, zorder=3)
        ax.add_patch(arrow)
        
        # Label position
        mx = (src_pt[0] + dst_pt[0]) / 2 + (rad * 8)
        my = (src_pt[1] + dst_pt[1]) / 2 + (rad * 8)
        
        lbl_box = FancyBboxPatch((mx - 7.5, my - 1.8), 15, 3.6, boxstyle="round,pad=0.2,rounding_size=0.5",
                                 facecolor='#FFFFFF', edgecolor=color, linewidth=0.8, zorder=5)
        ax.add_patch(lbl_box)
        ax.text(mx, my, rel_type, fontsize=7.5, weight='bold', ha='center', va='center',
                color=color, fontfamily='sans-serif', zorder=6)

    # Relationships between Use Cases
    # UC1 -> UC2 (flow / include)
    draw_relation("UC1", "UC2", "«include»", "#2563EB", rad=0.0)
    # UC2 -> UC3
    draw_relation("UC2", "UC3", "«include»", "#2563EB", rad=-0.12)
    # UC2 -> UC4
    draw_relation("UC2", "UC4", "«include»", "#2563EB", rad=0.0)
    # UC5 extends UC2
    draw_relation("UC5", "UC2", "«extend»", "#9333EA", rad=-0.15)
    # UC5 -> UC6
    draw_relation("UC5", "UC6", "«include»", "#9333EA", rad=0.0)
    # UC7 extends UC5
    draw_relation("UC7", "UC5", "«extend»", "#9333EA", rad=0.0)
    # UC8 -> UC9
    draw_relation("UC8", "UC9", "«include»", "#D97706", rad=0.0)
    # UC9 -> UC10
    draw_relation("UC9", "UC10", "«include»", "#D97706", rad=-0.1)
    # UC11 -> UC12
    draw_relation("UC11", "UC12", "«include»", "#16A34A", rad=0.0)
    # UC12 -> UC13
    draw_relation("UC12", "UC13", "«include»", "#16A34A", rad=0.0)

    # Legend Box
    leg_box = FancyBboxPatch((43, 8), 104, 6.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                             facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.2, zorder=2)
    ax.add_patch(leg_box)
    
    # Legend Items
    leg_items = [
        ("Inference & Detection", "#2563EB", "#EFF6FF"),
        ("Explainable AI (XAI)", "#9333EA", "#FAF5FF"),
        ("Data Quality & Splits", "#D97706", "#FFFBEB"),
        ("Training & Evaluation", "#16A34A", "#F0FDF4"),
        ("«include» / «extend» Dependency", "#64748B", "#FFFFFF")
    ]
    
    lx = 46
    for title, col, bg in leg_items:
        b = FancyBboxPatch((lx, 9.2), 3.5, 3.8, boxstyle="round,pad=0.2,rounding_size=0.6",
                           facecolor=bg, edgecolor=col, linewidth=1.2, zorder=3)
        ax.add_patch(b)
        ax.text(lx + 4.5, 11.1, title, fontsize=8, weight='bold', va='center', color='#334155', fontfamily='sans-serif', zorder=3)
        lx += 20.5

    plt.tight_layout()
    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"Saved Use Case Diagram to: {p}")
    plt.close()


def create_class_diagram(output_paths):
    fig, ax = plt.subplots(figsize=(22, 16), dpi=300)
    ax.set_xlim(0, 220)
    ax.set_ylim(0, 160)
    ax.axis('off')
    
    # Background
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    
    # Title Header
    ax.text(110, 154, "AI-Generated Image Detection & XAI — UML Class Diagram", 
            fontsize=22, weight='bold', ha='center', va='center', color='#0F172A', fontfamily='sans-serif')
    ax.text(110, 149.5, "Object-Oriented Architecture, Modules, Classifier Heads, XAI & Dataset Management", 
            fontsize=13, style='italic', ha='center', va='center', color='#475569', fontfamily='sans-serif')

    # Class Box Drawing Function (UML 3-compartment standard)
    def draw_uml_class(x, y, w, header_h, attr_h, meth_h, stereotype, class_name, attrs, methods, 
                       header_bg="#1E293B", body_bg="#FFFFFF", border_col="#334155"):
        total_h = header_h + attr_h + meth_h
        
        # Outer Card
        card = FancyBboxPatch((x, y - total_h), w, total_h, boxstyle="round,pad=0.3,rounding_size=1.2",
                              facecolor=body_bg, edgecolor=border_col, linewidth=1.8, zorder=2)
        ax.add_patch(card)
        
        # Header Box
        header = FancyBboxPatch((x, y - header_h), w, header_h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor=header_bg, edgecolor=border_col, linewidth=1.8, zorder=3)
        ax.add_patch(header)
        
        # Text Header
        if stereotype:
            ax.text(x + w/2, y - header_h*0.35, f"«{stereotype}»", fontsize=8.5, style='italic',
                    ha='center', va='center', color='#E2E8F0', fontfamily='sans-serif', zorder=4)
            ax.text(x + w/2, y - header_h*0.75, class_name, fontsize=11, weight='bold',
                    ha='center', va='center', color='#FFFFFF', fontfamily='sans-serif', zorder=4)
        else:
            ax.text(x + w/2, y - header_h*0.55, class_name, fontsize=12, weight='bold',
                    ha='center', va='center', color='#FFFFFF', fontfamily='sans-serif', zorder=4)
            
        # Divider 1 (between header & attrs)
        ax.plot([x, x + w], [y - header_h, y - header_h], color=border_col, lw=1.5, zorder=4)
        
        # Attributes List
        curr_y = y - header_h - 2.5
        for attr in attrs:
            ax.text(x + 1.8, curr_y, attr, fontsize=8.5, va='center', color='#1E293B', fontfamily='monospace', zorder=4)
            curr_y -= 2.4
            
        # Divider 2 (between attrs & methods)
        ax.plot([x, x + w], [y - header_h - attr_h, y - header_h - attr_h], color=border_col, lw=1.2, zorder=4)
        
        # Methods List
        curr_y = y - header_h - attr_h - 2.5
        for meth in methods:
            ax.text(x + 1.8, curr_y, meth, fontsize=8.5, va='center', color='#0F172A', fontfamily='monospace', zorder=4)
            curr_y -= 2.4
            
        return {
            "top": (x + w/2, y),
            "bottom": (x + w/2, y - total_h),
            "left": (x, y - total_h/2),
            "right": (x + w, y - total_h/2),
            "rect": (x, y - total_h, w, total_h)
        }

    classes = {}
    
    # 1. Base Framework / Parent Class: nn.Module
    classes["nn.Module"] = draw_uml_class(
        x=88, y=143, w=44, header_h=6, attr_h=4.5, meth_h=6.5,
        stereotype="PyTorch Core",
        class_name="torch.nn.Module",
        attrs=["- _parameters: OrderedDict", "- _modules: OrderedDict"],
        methods=["+ forward(*input): Tensor", "+ to(device): self", "+ train(mode=True): self"],
        header_bg="#334155", border_col="#1E293B"
    )

    # 2. End-to-End Model: DINOv2Classifier
    classes["DINOv2Classifier"] = draw_uml_class(
        x=12, y=118, w=62, header_h=7, attr_h=16, meth_h=10,
        stereotype="Neural Network Module",
        class_name="DINOv2Classifier",
        attrs=[
            "+ backbone_name: str = 'dinov2_vits14'",
            "+ freeze_backbone: bool = True",
            "+ backbone: VisionTransformer",
            "+ embed_dim: int = 384",
            "+ classifier_type: str ('linear' | 'mlp')",
            "+ classifier: nn.Module"
        ],
        methods=[
            "+ __init__(backbone_name, classifier_type, ...)",
            "+ forward(x: Tensor): Tensor",
            "+ get_embedding_dimension(): int"
        ],
        header_bg="#1E40AF", border_col="#1D4ED8"
    )

    # 3. LinearClassifier Head
    classes["LinearClassifier"] = draw_uml_class(
        x=82, y=118, w=54, header_h=7, attr_h=8, meth_h=8,
        stereotype="Classification Head",
        class_name="LinearClassifier",
        attrs=[
            "+ fc: nn.Linear(384, 2)"
        ],
        methods=[
            "+ __init__(embed_dim=384, num_classes=2)",
            "+ forward(x: Tensor): Tensor"
        ],
        header_bg="#0D9488", border_col="#0F766E"
    )

    # 4. MLPClassifier Head
    classes["MLPClassifier"] = draw_uml_class(
        x=144, y=118, w=64, header_h=7, attr_h=11, meth_h=8,
        stereotype="Classification Head",
        class_name="MLPClassifier",
        attrs=[
            "+ hidden_dim: int = 256",
            "+ dropout_rate: float = 0.3",
            "+ mlp: nn.Sequential [Linear, ReLU, Drop, Lin]"
        ],
        methods=[
            "+ __init__(embed_dim=384, num_classes=2, ...)",
            "+ forward(x: Tensor): Tensor"
        ],
        header_bg="#047857", border_col="#065F46"
    )

    # 5. Dataset Class: AIDetectionDataset
    classes["AIDetectionDataset"] = draw_uml_class(
        x=12, y=72, w=62, header_h=7, attr_h=10, meth_h=12,
        stereotype="PyTorch Dataset",
        class_name="AIDetectionDataset",
        attrs=[
            "- df: pd.DataFrame",
            "- transform: torchvision.transforms.Compose"
        ],
        methods=[
            "+ __init__(csv_file: str, transform=None)",
            "+ __len__(): int",
            "+ __getitem__(idx: int): Tuple[Tensor, int]",
            "+ get_dinov2_transforms(img_size=224): Compose"
        ],
        header_bg="#D97706", border_col="#B45309"
    )

    # 6. DSU Class for Splitting
    classes["DSU"] = draw_uml_class(
        x=82, y=72, w=54, header_h=7, attr_h=8, meth_h=11,
        stereotype="Data Structure",
        class_name="DSU (Disjoint Set Union)",
        attrs=[
            "- parent: Dict[str, str]"
        ],
        methods=[
            "+ __init__()",
            "+ find(x: str): str",
            "+ union(x: str, y: str): None"
        ],
        header_bg="#B45309", border_col="#78350F"
    )

    # 7. XAIEngine Class (representing xai.py)
    classes["XAIEngine"] = draw_uml_class(
        x=144, y=72, w=64, header_h=7, attr_h=8, meth_h=17,
        stereotype="Explainability Module",
        class_name="XAIEngine",
        attrs=[
            "+ attn_input: Tensor (Captured Hook)",
            "+ patch_size: int = 14",
            "+ grid_size: int = 16"
        ],
        methods=[
            "+ attention_hook_fn(module, input, output)",
            "+ compute_attention_map(model, img, dev)",
            "+ generate_heatmap_overlay(orig_img, attn)",
            "+ run_faithfulness_test(path, model, dev)",
            "+ save_xai_visualization(path, model, dev)"
        ],
        header_bg="#7E22CE", border_col="#6B21A8"
    )

    # 8. ModelTrainer & Evaluator (Service Layer)
    classes["ModelTrainer"] = draw_uml_class(
        x=12, y=28, w=62, header_h=7, attr_h=11, meth_h=14,
        stereotype="Training Pipeline",
        class_name="ModelTrainer",
        attrs=[
            "+ criterion: nn.CrossEntropyLoss",
            "+ optimizer: torch.optim.Adam",
            "+ early_stopping_patience: int = 10"
        ],
        methods=[
            "+ train_classifier(args): None",
            "+ set_seed(seed=42): None",
            "+ evaluate_metrics(y_true, y_pred, y_prob)",
            "+ checkpoint_model(f1, state_dict): None"
        ],
        header_bg="#0284C7", border_col="#0369A1"
    )

    # 9. ModelEvaluator
    classes["ModelEvaluator"] = draw_uml_class(
        x=82, y=28, w=54, header_h=7, attr_h=8, meth_h=14,
        stereotype="Evaluation & Benchmarking",
        class_name="ModelEvaluator",
        attrs=[
            "+ metrics_summary: Dict[str, float]",
            "+ report_path: str"
        ],
        methods=[
            "+ evaluate_on_test_set(path, split): Dict",
            "+ plot_confusion_matrix(y_true, preds)",
            "+ plot_roc_curve(y_true, probs)",
            "+ export_classification_report(): None"
        ],
        header_bg="#0E7490", border_col="#155E75"
    )

    # 10. Streamlit Dashboard App
    classes["StreamlitDashboard"] = draw_uml_class(
        x=144, y=28, w=64, header_h=7, attr_h=8, meth_h=14,
        stereotype="Interactive GUI",
        class_name="StreamlitDashboardApp",
        attrs=[
            "+ page_title: str",
            "+ device: str ('cuda' | 'cpu')"
        ],
        methods=[
            "+ load_detection_model(model_path, dev)",
            "+ render_sidebar(): None",
            "+ process_upload(file): None",
            "+ render_prediction_and_xai(orig, overlay)"
        ],
        header_bg="#15803D", border_col="#166534"
    )

    # UML Relationship Connectors Helper
    def draw_inheritance(sub_class, super_class, color="#334155"):
        start = classes[sub_class]["top"]
        end = classes[super_class]["bottom"]
        
        # Path
        ax.plot([start[0], start[0]], [start[1], start[1] + 3], color=color, lw=1.5, zorder=3)
        ax.plot([start[0], end[0]], [start[1] + 3, end[1] - 3], color=color, lw=1.5, zorder=3)
        ax.plot([end[0], end[0]], [end[1] - 3, end[1]], color=color, lw=1.5, zorder=3)
        
        # Closed hollow triangle head
        tri = Polygon([[end[0], end[1]], [end[0] - 1.6, end[1] - 2.8], [end[0] + 1.6, end[1] - 2.8]],
                      closed=True, facecolor="#FFFFFF", edgecolor=color, lw=1.5, zorder=4)
        ax.add_patch(tri)

    def draw_association(cls_a, cls_b, label="", color="#475569", style="-", rad=0.0, head=""):
        start = classes[cls_a]["right"] if classes[cls_a]["right"][0] < classes[cls_b]["left"][0] else classes[cls_a]["bottom"]
        end = classes[cls_b]["left"] if classes[cls_a]["right"][0] < classes[cls_b]["left"][0] else classes[cls_b]["top"]
        
        if head == "composition":
            # Filled Diamond at start
            dia = Polygon([[start[0], start[1]], [start[0] + 1.5, start[1] + 1.2], 
                           [start[0] + 3.0, start[1]], [start[0] + 1.5, start[1] - 1.2]],
                          closed=True, facecolor="#1E293B", edgecolor="#1E293B", lw=1.2, zorder=4)
            ax.add_patch(dia)
            ax.plot([start[0] + 3.0, end[0]], [start[1], end[1]], color=color, lw=1.5, linestyle=style, zorder=3)
        elif head == "aggregation":
            # Hollow Diamond at start
            dia = Polygon([[start[0], start[1]], [start[0] + 1.5, start[1] + 1.2], 
                           [start[0] + 3.0, start[1]], [start[0] + 1.5, start[1] - 1.2]],
                          closed=True, facecolor="#FFFFFF", edgecolor="#1E293B", lw=1.2, zorder=4)
            ax.add_patch(dia)
            ax.plot([start[0] + 3.0, end[0]], [start[1], end[1]], color=color, lw=1.5, linestyle=style, zorder=3)
        else:
            arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}",
                                    linestyle=style, linewidth=1.4, color=color,
                                    arrowstyle="-|>", mutation_scale=12, zorder=3)
            ax.add_patch(arrow)
            
        if label:
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2
            ax.text(mx, my + 1.5, label, fontsize=8, weight='bold', color=color,
                    ha='center', va='center', backgroundcolor='#FFFFFF', zorder=5)

    # Draw Inheritances
    draw_inheritance("DINOv2Classifier", "nn.Module", "#1D4ED8")
    draw_inheritance("LinearClassifier", "nn.Module", "#0F766E")
    draw_inheritance("MLPClassifier", "nn.Module", "#065F46")

    # Associations / Compositions
    # DINOv2Classifier has LinearClassifier or MLPClassifier as head
    draw_association("DINOv2Classifier", "LinearClassifier", label="«aggregates head»", color="#2563EB", head="aggregation")
    
    # ModelTrainer uses AIDetectionDataset & Classifiers
    draw_association("ModelTrainer", "AIDetectionDataset", label="«loads samples»", color="#D97706", style="--")
    
    # ModelTrainer produces Checkpoints used by ModelEvaluator
    draw_association("ModelTrainer", "ModelEvaluator", label="«checkpoint»", color="#0284C7", style="-")
    
    # XAIEngine hooks into DINOv2Classifier
    draw_association("XAIEngine", "MLPClassifier", label="«explains predictions»", color="#7E22CE", style="--")
    
    # StreamlitDashboard uses DINOv2Classifier and XAIEngine
    draw_association("StreamlitDashboard", "XAIEngine", label="«visualizes»", color="#15803D", style="--")
    draw_association("StreamlitDashboard", "ModelEvaluator", label="«displays metrics»", color="#15803D", style="--")

    # Legend / UML Notation Summary
    leg_box = FancyBboxPatch((12, 1.5), 196, 5.5, boxstyle="round,pad=0.3,rounding_size=0.8",
                             facecolor='#F1F5F9', edgecolor='#CBD5E1', linewidth=1.2, zorder=2)
    ax.add_patch(leg_box)
    
    ax.text(14, 4.25, "UML Notation Key:", fontsize=8.5, weight='bold', color='#0F172A', va='center')
    ax.text(38, 4.25, "──▷  Generalization / Inheritance (is-a)", fontsize=8, color='#334155', va='center')
    ax.text(82, 4.25, "◇──  Aggregation (has-a head)", fontsize=8, color='#334155', va='center')
    ax.text(122, 4.25, "- - ▷  Dependency / Uses relationship", fontsize=8, color='#334155', va='center')
    ax.text(168, 4.25, "+ Public   - Private attribute/method", fontsize=8, color='#334155', va='center')

    plt.tight_layout()
    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        plt.savefig(p, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"Saved Class Diagram to: {p}")
    plt.close()


if __name__ == "__main__":
    snapshot_dirs = [
        "e:/PROJECT/AI-Image-Detection/review 2 snapshot",
        "e:/PROJECT/AI-Image-Detection/review 2 snaphot"
    ]
    
    uc_paths = [os.path.join(d, "use_case_diagram.png") for d in snapshot_dirs]
    class_paths = [os.path.join(d, "class_diagram.png") for d in snapshot_dirs]
    
    print("Generating Use Case Diagram...")
    create_use_case_diagram(uc_paths)
    
    print("Generating UML Class Diagram...")
    create_class_diagram(class_paths)
    print("All diagrams generated successfully!")
