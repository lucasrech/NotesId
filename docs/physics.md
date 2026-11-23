# Física da Corda Rígida e Inharmonicidade

Este documento descreve o modelo físico utilizado pelo sistema para distinguir posições `(corda, casa)` que produzem a **mesma frequência fundamental** $f_0$ (por exemplo, o Dó4 ≈ 261,63 Hz em até 5 cordas diferentes).

## 1. Equação de onda de uma corda rígida

Uma corda ideal perfeitamente flexível de comprimento vibrante $L$, tensão $T$ e densidade linear de massa $\mu$ obedece à equação de onda clássica e possui parciais exatamente harmônicos $f_n = n f_0$, com:

$$f_0 = \frac{1}{2L}\sqrt{\frac{T}{\mu}}$$

Uma corda real de guitarra possui **rigidez à flexão** $EI$ (onde $E$ é o módulo de Young do aço e $I$ é o momento de inércia de área da seção transversal). A equação de onda passa a incluir um termo dispersivo de quarta ordem (modelo de viga de Euler-Bernoulli sob tensão):

$$\mu \frac{\partial^2 y}{\partial t^2} = T \frac{\partial^2 y}{\partial x^2} - EI \frac{\partial^4 y}{\partial x^4}$$

Para extremidades simplesmente apoiadas (aproximação padrão para pestana/traste e ponte), as frequências próprias dos modos normais deixam de ser múltiplos inteiros de $f_0$ e seguem a lei de inharmonicidade:

$$f_n = n f_0 \sqrt{1 + B n^2}$$

onde $B$ é o **coeficiente de inharmonicidade** (adimensional, tipicamente entre $2 \times 10^{-5}$ e $2 \times 10^{-3}$ em guitarras elétricas):

$$B = \frac{\pi^2 E I}{T L^2}$$

Para uma corda lisa cilíndrica de diâmetro $d$, $I = \frac{\pi d^4}{64}$, resultando em:

$$B = \frac{\pi^3 E d^4}{64 T L^2}$$

## 2. Variação de $B$ ao longo da mesma corda (em função da casa)

Ao pressionar a mesma corda na casa $k \in \{0, 1, \dots, 22\}$, o comprimento vibrante encurta conforme o temperamento igual:

$$L(k) = L_0 \cdot 2^{-k/12}$$

onde $L_0$ é o comprimento de escala (ex.: 648 mm em uma Stratocaster). Mantendo a tensão $T$ aproximadamente constante:

$$B \propto \frac{1}{L^2} \implies B(k) = B_0 \cdot \left(\frac{L_0}{L(k)}\right)^2 = B_0 \cdot 2^{k/6}$$

Cada oitava (+12 casas) reduz o comprimento à metade e **quadruplica** o coeficiente $B$ ($2^{12/6} = 4$).

Essa lei permite calibrar a guitarra medindo $B$ em poucas casas conhecidas (por exemplo, corda solta `k = 0` e casa `k = 12`), estimar $B_0$ por corda e prever $B(k)$ em **todas** as demais casas sem precisar gravar o braço inteiro.

## 3. Comparação entre cordas diferentes na mesma nota ($f_0$ fixa)

Substituindo $T = 4 L^2 f_0^2 \mu$ na definição de $B$, obtemos a relação para a mesma frequência fundamental $f_0$ tocada em cordas diferentes:

$$B = \frac{\pi^2 E}{4} \cdot \frac{I / \mu}{L^4 f_0^2} \propto \frac{I / \mu}{L^4 f_0^2}$$

Para uma mesma nota (mesma $f_0$), tocá-la em uma corda mais grave exige uma casa mais alta (comprimento $L$ bem menor). Como $B$ cresce com $1/L^4$, cordas mais graves tocadas em casas altas apresentam inharmonicidade drasticamente maior.

### O cruzamento entre corda 3 (lisa) e corda 4 (encapada)

Em encordoamentos padrão de guitarra elétrica (ex.: `.009–.042` ou `.010–.046`):
- As cordas **1, 2 e 3** (E4, B3, G3) são de aço sólido **liso** (*plain*), onde $I \propto d^4$.
- As cordas **4, 5 e 6** (D3, A2, E2) são **encapadas** (*wound*): possuem um núcleo interno fino de aço envolto por um fio espiralado. O enrolamento aumenta $\mu$ significativamente, mas contribui muito pouco para a rigidez à flexão $EI$, que continua dominada pelo núcleo fino.

Consequentemente, na Fender Stratocaster do EGFxSet, a corda 3 (G3 lisa, $B_0 \approx 1{,}14 \times 10^{-4}$) é **mais rígida** que a corda 4 (D3 encapada, $B_0 \approx 6{,}60 \times 10^{-5}$). Ao tocar a mesma nota na casa $k$ da corda 3 versus casa $k + 5$ da corda 4 (por exemplo, Dó4 em `(3, 5)` vs `(4, 10)`), os valores de $B$ quase coincidem ($\approx 2{,}0 \times 10^{-4}$). Por isso, o sistema combina $B$ com características complementares e permite a saída `indeterminado` quando há empate físico.

## 4. Linearização e ajuste robusto de $B$

Elevando ao quadrado a relação $f_n = n f_0 \sqrt{1 + B n^2}$ e dividindo por $n^2$:

$$\left(\frac{f_n}{n}\right)^2 = f_0^2 + (f_0^2 B) \cdot n^2$$

Trata-se de uma regressão linear exata de $y_n = (f_n / n)^2$ em função de $x_n = n^2$, com intercepto $c_0 = f_0^2$ e inclinação $c_1 = f_0^2 B$, logo $B = c_1 / c_0$ e $f_{0,\text{ref}} = \sqrt{c_0}$. O sistema resolve essa regressão por **mínimos quadrados iterativamente reponderados (IRLS com pesos de Huber)**, o que torna a estimativa robusta a parciais ruidosos ou suprimidos por nulos de palhetada/captador.

## 5. Características físicas complementares

1. **Taxa de decaimento por parcial (STFT):** perdas viscoelásticas no material da corda, atrito com o ar e perdas no traste/dedo fazem parciais agudos e cordas curtas/encapadas decaírem mais rápido (em dB/s).
2. **Posição relativa de palhetada ($p$):** pinçar a corda na fração $p$ do seu comprimento suprime parciais próximos a $n = k / p$ (fator $|\sin(n \pi p)|$).
3. **Centroide e roll-off espectral:** descritores globais de brilho usados apenas com peso reduzido no classificador.
