<!--
[![CI](https://github.com/SEU_USUARIO/guitar-string-identifier/actions/workflows/ci.yml/badge.svg)](https://github.com/SEU_USUARIO/guitar-string-identifier/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
-->

# Identificador de Corda e Casa de Guitarra

Sistema para identificar **qual corda e qual casa** foram tocadas em uma guitarra a partir do áudio de uma nota isolada, usando **apenas algoritmos clássicos de processamento de sinais** (sem machine learning).

A diferenciação entre posições que produzem a mesma frequência fundamental (como o Dó4 em até 5 cordas diferentes) é feita explorando propriedades físicas da corda, principalmente o **coeficiente de inharmonicidade $B$**, complementado por taxas de decaimento por parcial e características espectrais.

> **Status:** Estrutura inicial criada. Documentação completa, resultados de avaliação e instruções de execução serão atualizados ao longo do desenvolvimento.
