"""
run_all_analysis.py
-------------------
Orquestrador que roda os três scripts de análise em sequência e gera
um relatório HTML consolidado com os gráficos embutidos.

Saídas
------
  results/dataset_analysis_images.json   / .png
  results/dataset_analysis_embeddings.json / .png
  results/dataset_analysis_shards.json   / .png
  results/report.html                    ← relatório consolidado

Uso
---
  python data/run_all_analysis.py
  python data/run_all_analysis.py --skip_images   # pula análise de imagens (pesada)
  python data/run_all_analysis.py --skip_embeds   # pula análise de embeddings
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Imports locais — importados como módulos para não duplicar código
import analyze_images as _img_mod
import analyze_embeddings as _emb_mod
import analyze_shards as _shard_mod


# ---------------------------------------------------------------------------
# Geração do relatório HTML
# ---------------------------------------------------------------------------

def _img_to_base64(path: str) -> str:
    """
    Converte imagem para string base64 para embutir no HTML.

    Parameters
    ----------
    path:
        Caminho do arquivo PNG.

    Returns
    -------
    str com a imagem em base64.
    """
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_html_report(output_dir: str) -> str:
    """
    Gera relatório HTML consolidando os JSONs e PNGs produzidos.

    Parameters
    ----------
    output_dir:
        Diretório com os arquivos de resultados.

    Returns
    -------
    str com caminho do HTML gerado.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        ("Imagens e Captions",  "dataset_analysis_images"),
        ("Embeddings H5",       "dataset_analysis_embeddings"),
        ("Estrutura dos Shards","dataset_analysis_shards"),
    ]

    html_sections = ""
    for title, prefix in sections:
        json_path = os.path.join(output_dir, f"{prefix}.json")
        png_path  = os.path.join(output_dir, f"{prefix}.png")

        # Sumário JSON (filtra listas longas)
        json_html = ""
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Filtra listas de detalhes por shard para não poluir
            data_clean = {k: v for k, v in data.items()
                          if not isinstance(v, list) or k in ("splits",)}
            json_html = f"<pre style='background:#1e1e2e;color:#cdd6f4;padding:12px;"
            json_html += f"border-radius:6px;overflow:auto;font-size:12px;'>"
            json_html += json.dumps(data_clean, indent=2, ensure_ascii=False)
            json_html += "</pre>"

        # Imagem base64
        img_html = ""
        if os.path.exists(png_path):
            b64 = _img_to_base64(png_path)
            img_html = (f"<img src='data:image/png;base64,{b64}' "
                        f"style='max-width:100%;border-radius:8px;margin-top:12px;'/>")

        html_sections += f"""
        <section style='margin-bottom:48px;'>
          <h2 style='color:#89b4fa;border-bottom:1px solid #45475a;padding-bottom:6px;'>{title}</h2>
          {img_html}
          <details style='margin-top:12px;'>
            <summary style='cursor:pointer;color:#a6e3a1;'>Dados JSON</summary>
            {json_html}
          </details>
        </section>
        """

    html = f"""<!DOCTYPE html>
<html lang='pt-BR'>
<head>
  <meta charset='UTF-8'>
  <meta name='viewport' content='width=device-width,initial-scale=1'>
  <title>COYO Dataset Analysis</title>
  <style>
    body  {{ font-family: 'Segoe UI', sans-serif; background:#1e1e2e; color:#cdd6f4;
             max-width:1200px; margin:0 auto; padding:32px 24px; }}
    h1    {{ color:#cba6f7; }}
    h2    {{ color:#89b4fa; }}
    summary {{ font-weight: bold; }}
  </style>
</head>
<body>
  <h1>🔬 COYO Dataset — Relatório de Análise</h1>
  <p style='color:#a6adc8;'>Gerado em: {ts}</p>
  {html_sections}
</body>
</html>"""

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHTML report saved to: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all(
    root_images: str,
    max_images: int,
    shard_val: str,
    max_embeds: int,
    train_dir: str,
    val_dir:   str,
    test_dir:  str,
    output_dir: str,
    skip_images: bool,
    skip_embeds: bool,
    skip_shards: bool,
) -> None:
    """
    Executa os três scripts de análise e gera relatório consolidado.

    Parameters
    ----------
    root_images:
        Diretório com imagens extraídas (para analyze_images).
    max_images:
        Número máximo de imagens a analisar.
    shard_val:
        Diretório dos shards de validação (para analyze_embeddings).
    max_embeds:
        Número máximo de embeddings a amostrar.
    train_dir, val_dir, test_dir:
        Diretórios dos shards para analyze_shards.
    output_dir:
        Diretório de saída para todos os resultados.
    skip_images, skip_embeds, skip_shards:
        Flags para pular análises específicas.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not skip_images:
        print("\n" + "─" * 55)
        print("1/3 Image and Caption Analysis")
        print("─" * 55)
        _img_mod.analyze(
            root_dir=root_images,
            max_samples=max_images,
            output_dir=output_dir,
            top_words=30,
        )

    if not skip_embeds:
        print("\n" + "─" * 55)
        print("2/3 H5 Embeddings Analysis")
        print("─" * 55)
        _emb_mod.analyze(
            shard_dir=shard_val,
            max_samples=max_embeds,
            output_dir=output_dir,
            n_neg_pairs=10_000,
            n_heatmap=100,
        )

    if not skip_shards:
        print("\n" + "─" * 55)
        print("3/3 Shard Structure Analysis")
        print("─" * 55)
        _shard_mod.analyze(
            train_dir=train_dir or None,
            val_dir=val_dir     or None,
            test_dir=test_dir   or None,
            output_dir=output_dir,
        )

    print("\n" + "─" * 55)
    print("Generating consolidated HTML report...")
    _build_html_report(output_dir)
    print("─" * 55)
    print("✓ Analysis complete!")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orquestra análise completa do COYO.")
    parser.add_argument("--root_images",   default="F:/COYO/coyo/extracted")
    parser.add_argument("--max_images",    type=int, default=480000)
    parser.add_argument("--shard_val",     default="G:/coyo/embeds/val_anyup")
    parser.add_argument("--max_embeds",    type=int, default=5000)
    parser.add_argument("--train_dir",     default="F:/COYO/embeds/train_anyup")
    parser.add_argument("--val_dir",       default="G:/coyo/embeds/val_anyup")
    parser.add_argument("--test_dir",      default="G:/coyo/embeds/test_anyup")
    parser.add_argument("--output_dir",    default="results")
    parser.add_argument("--skip_images",   action="store_true")
    parser.add_argument("--skip_embeds",   action="store_true")
    parser.add_argument("--skip_shards",   action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_all(
        root_images=args.root_images,
        max_images=args.max_images,
        shard_val=args.shard_val,
        max_embeds=args.max_embeds,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        skip_images=args.skip_images,
        skip_embeds=args.skip_embeds,
        skip_shards=args.skip_shards,
    )
