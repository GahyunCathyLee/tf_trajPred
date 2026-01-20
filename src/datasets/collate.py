from typing import Any, Dict, List
import torch

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x_ego = torch.stack([b["x_ego"] for b in batch], dim=0)        
    x_nb = torch.stack([b["x_nb"] for b in batch], dim=0)          
    nb_mask = torch.stack([b["nb_mask"] for b in batch], dim=0)    
    y = torch.stack([b["y"] for b in batch], dim=0)                
    x_last_abs = torch.stack([b["x_last_abs"] for b in batch], dim=0)

    out: Dict[str, Any] = {
        "x_ego": x_ego,
        "x_nb": x_nb,
        "nb_mask": nb_mask,
        "y": y,
        "x_last_abs": x_last_abs,
    }

    if "style_prob" in batch[0]:
        out["style_prob"] = torch.stack([b["style_prob"] for b in batch], dim=0)
        out["style_valid"] = torch.stack([b["style_valid"] for b in batch], dim=0)
        if "style_label" in batch[0]:
            out["style_label"] = torch.stack([b["style_label"] for b in batch], dim=0)

    if "meta" in batch[0]:
        out["meta"] = [b.get("meta", {}) for b in batch]

    return out