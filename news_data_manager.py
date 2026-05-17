import json
import os
import re
from bs4 import BeautifulSoup


class NewsDataManager:
    METADATA_PATTERNS = re.compile(
        r'Publicado em:\s*[\d/]+\s*(?:às|as)\s*\d{2}h\d{2}'
        r'|\d{2}/\d{2}/\d{4}\s*[-—|]\s*\d{2}h\d{2}(?:\s*\|[^<\n]*)?'
        r'|\d{2}/\d{2}/\d{4}[^\S\n]*—[^\S\n]*\w+(?:[^\S\n]+\w+){0,3}'
        r'|\d{2}/\d{2}/\d{4}\s*(?:às|as)\s*\d{2}h\d{2}'
        r'|Publicado em:\s*\*{0,2}\d{2}/\d{2}/\d{4}\*{0,2}'
        r'|^\s*\d{2}/\d{2}/\d{4}\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    MIN_TOKENS = 20

    def __init__(self, raw_path: str, clean_path: str = "dados/noticias_limpas.json"):
        self.raw_path = raw_path
        self.clean_path = clean_path
        self.discarded: list[int] = []
        self.data: list[dict] = []

        if self._is_clean():
            print("Dados já limpos encontrados, carregando...")
            self._load_clean()
        else:
            print("Executando limpeza...")
            with open(self.raw_path, encoding="utf-8") as f:
                raw = json.load(f)
            self.data = self._clean_all(raw)
            self._save_clean()
            print(f"Aprovados: {len(self.data)} | Descartados: {self.discarded}")

    def _is_clean(self) -> bool:
        return os.path.exists(self.clean_path)

    def _load_clean(self) -> None:
        with open(self.clean_path, encoding="utf-8") as f:
            self.data = json.load(f)

    def _save_clean(self) -> None:
        os.makedirs(os.path.dirname(self.clean_path), exist_ok=True)
        with open(self.clean_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def _clean(self, text: str) -> str | None:
        text = BeautifulSoup(text, "html.parser").get_text()
        text = text.replace("\xa0", " ")
        text = self.METADATA_PATTERNS.sub("", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        text = text.strip()
        if len(text.split()) < self.MIN_TOKENS:
            return None
        return text

    def _clean_all(self, raw: list[dict]) -> list[dict]:
        cleaned = []
        for notice in raw:
            result = self._clean(notice["texto"])
            if result:
                cleaned.append({**notice, "texto": result})
            else:
                self.discarded.append(notice["id"])
        return cleaned

    def display(self) -> None:
        for notice in self.data:
            print(f"ID: {notice['id']}")
            print(f"Título: {notice['titulo']}")
            print(notice['texto'])
            print("-" * 50)