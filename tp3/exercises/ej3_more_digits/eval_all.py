"""Evalúa TODOS los modelos guardados en outputs/ej3_more_digits/models/
sobre digits_test.csv y los agrupa segun las tecnicas de regularizacion
declaradas en su config.

Uso:
    uv run python -m exercises.ej3_more_digits.eval_all
"""
import json
from pathlib import Path

import numpy as np

from common.datasets import load_digits_test, to_one_hot
from common.mlp import MLP


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "outputs" / "ej3_more_digits" / "models"
CONFIGS_DIRS = [
    ROOT / "configs" / "ej3_more_digits" / "baselines",
    ROOT / "configs" / "ej3_more_digits" / "historical",
    ROOT / "configs" / "ej3_more_digits" / "vanilla",
    ROOT / "configs" / "ej3_more_digits" / "ensembles" / "aug_variations",
    ROOT / "configs" / "ej3_more_digits" / "ensembles" / "diverse_architectures",
    ROOT / "configs" / "ej3_more_digits" / "ensembles" / "wd_variations",
]


def find_config_for_model(model_name: str):
    """Busca el .json cuyo save_model produce model_name.

    Para multi-seed: el config dice 'foo.npz' pero el archivo real es
    'foo_seedN.npz' (un .npz por seed). Acepta ambos casos.
    """
    base = model_name.replace(".npz", "")
    # quitar sufijo _seedN si existe
    base_no_seed = base
    if "_seed" in base:
        base_no_seed = base.rsplit("_seed", 1)[0]

    for d in CONFIGS_DIRS:
        for cfg_path in d.glob("*.json"):
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                continue
            sm = cfg.get("save_model", "")
            sm_base = Path(sm).name.replace(".npz", "") if sm else ""
            if sm_base == base or sm_base == base_no_seed:
                return cfg, cfg_path
    return None, None


def techniques(cfg: dict | None) -> list[str]:
    """Lista de tecnicas de regularizacion declaradas en el config."""
    if cfg is None:
        return ["?"]
    techs = []
    if cfg.get("weight_decay", 0.0) and cfg.get("weight_decay", 0.0) > 0:
        techs.append(f"L2(λ={cfg['weight_decay']})")
    if cfg.get("data_augmentation", False):
        techs.append("Aug")
    if cfg.get("patience"):
        techs.append(f"ES(p={cfg['patience']})")
    if cfg.get("lr_scheduler"):
        techs.append(f"LR-{cfg['lr_scheduler'].get('type', '?')}")
    return techs


def main():
    X_test, y_test = load_digits_test()
    Y_test = to_one_hot(y_test, 10, encoding="zero_one")

    rows = []
    for model_path in sorted(MODELS_DIR.glob("*.npz")):
        name = model_path.name
        cfg, cfg_path = find_config_for_model(name)
        try:
            mlp = MLP.load(str(model_path))
            m = mlp.evaluate(X_test, Y_test)
            acc = float(m["accuracy"])
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            continue
        rows.append({
            "name": name.replace(".npz", ""),
            "acc": acc,
            "techs": techniques(cfg),
            "arch": cfg.get("architecture") if cfg else None,
        })

    # Agrupar
    no_reg = [r for r in rows if not r["techs"]]
    with_reg = [r for r in rows if r["techs"]]
    above_98 = [r for r in rows if r["acc"] >= 0.98]

    def fmt(r):
        techs_s = " + ".join(r["techs"]) if r["techs"] else "—"
        arch_s = "x".join(map(str, r["arch"])) if r["arch"] else "?"
        return f"  {r['acc']*100:6.2f}%  {r['name']:<35} [{techs_s}]  arch={arch_s}"

    print("=" * 80)
    print(f"EVAL DE {len(rows)} MODELOS DE EJ3 SOBRE digits_test.csv ({len(y_test)} muestras)")
    print("=" * 80)

    print(f"\n--- SIN regularizacion ({len(no_reg)}) ---")
    for r in sorted(no_reg, key=lambda x: -x["acc"]):
        print(fmt(r))

    print(f"\n--- CON regularizacion ({len(with_reg)}) ---")
    for r in sorted(with_reg, key=lambda x: -x["acc"]):
        print(fmt(r))

    print(f"\n--- ACCURACY >= 98% ({len(above_98)}) ---")
    for r in sorted(above_98, key=lambda x: -x["acc"]):
        print(fmt(r))

    # Tambien CSV resumen
    out_csv = ROOT / "outputs" / "ej3_more_digits" / "metrics" / "all_models_eval.csv"
    with open(out_csv, "w") as f:
        f.write("model,test_accuracy,techniques,architecture\n")
        for r in sorted(rows, key=lambda x: -x["acc"]):
            techs_s = "|".join(r["techs"]) if r["techs"] else ""
            arch_s = "x".join(map(str, r["arch"])) if r["arch"] else ""
            f.write(f'{r["name"]},{r["acc"]:.4f},{techs_s},{arch_s}\n')
    print(f"\nGuardado: {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
