"""Leitura, validação e gravação de perfis YAML em configs/profiles/."""

from __future__ import annotations

from pathlib import Path

import yaml

from guitar_string_identifier.profiles.schema import GuitarProfile


class ProfileStore:
    """Gerencia CRUD de perfis YAML validados por Pydantic."""

    def __init__(self, profiles_dir: Path | str = "configs/profiles") -> None:
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def _profile_path(self, name: str) -> Path:
        safe = name.strip()
        if not safe or "/" in safe or "\\" in safe or ".." in safe:
            raise ValueError(f"Nome de perfil inválido: {name!r}")
        return self.profiles_dir / f"{safe}.yaml"

    def list_profiles(self) -> list[str]:
        """Retorna os nomes dos perfis disponíveis ordenados alfabeticamente."""
        return sorted(p.stem for p in self.profiles_dir.glob("*.yaml"))

    def exists(self, name: str) -> bool:
        return self._profile_path(name).exists()

    def load(self, name: str) -> GuitarProfile:
        path = self._profile_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Perfil '{name}' não encontrado em {path}.")
        with path.open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp) or {}
        raw.setdefault("name", name)
        return GuitarProfile.model_validate(raw)

    def save(self, profile: GuitarProfile) -> Path:
        path = self._profile_path(profile.name)
        data = profile.model_dump(mode="python")
        with path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(data, fp, sort_keys=False, allow_unicode=True)
        return path

    def duplicate(self, source_name: str, target_name: str) -> GuitarProfile:
        if self.exists(target_name):
            raise FileExistsError(f"Perfil '{target_name}' já existe.")
        profile = self.load(source_name)
        cloned = profile.model_copy(update={"name": target_name})
        self.save(cloned)
        return cloned

    def rename(self, old_name: str, new_name: str) -> GuitarProfile:
        if old_name == new_name:
            return self.load(old_name)
        if self.exists(new_name):
            raise FileExistsError(f"Perfil '{new_name}' já existe.")
        profile = self.load(old_name)
        renamed = profile.model_copy(update={"name": new_name})
        self.save(renamed)
        self._profile_path(old_name).unlink(missing_ok=True)
        return renamed

    def delete(self, name: str) -> None:
        path = self._profile_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Perfil '{name}' não encontrado.")
        path.unlink()
