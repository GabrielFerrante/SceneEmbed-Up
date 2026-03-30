from pathlib import Path
import os

root = Path(r"F:/COYO/embeds/train_anyup")

print("=== Estrutura do diretório ===")
for item in sorted(root.iterdir()):
    print(f"  {item.name}  ({'pasta' if item.is_dir() else 'arquivo'})")
    if item.is_dir():
        filhos = sorted(item.iterdir())[:5]  # mostra só os 5 primeiros
        for f in filhos:
            print(f"    {f.name}")