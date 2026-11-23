# Contribuindo

Obrigado pelo interesse em contribuir com o **Guitar String Identifier**!

## Padrões do projeto

- **Idioma**: documentação e textos de interface em português do Brasil (`pt-BR`); nomes de código, módulos, funções e variáveis em inglês.
- **Sem machine learning**: o núcleo de identificação deve permanecer baseado exclusivamente em algoritmos clássicos de processamento de sinais e física acústica.
- **Commits**: utilize o padrão [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`).

## Fluxo de trabalho

1. Crie um ambiente local com `uv sync`.
2. Faça suas alterações em uma branch a partir de `main`.
3. Rode lint e testes antes de abrir PR:
   ```bash
   uv run ruff check .
   uv run pytest
   ```
4. Atualize `requirements.txt` se alterar dependências:
   ```bash
   uv export --no-hashes --format requirements-txt -o requirements.txt
   ```
