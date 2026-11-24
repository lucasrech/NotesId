# Glossário — NotesId

> 🇺🇸 [English version](GLOSSARY.md) | 🇧🇷 **Português**

Este glossário explica os termos musicais, físicos e de processamento de sinais usados no **NotesId** em linguagem simples, sem exigir formação prévia em música ou engenharia.

---

## Termos musicais básicos

### Nota
O nome dado a uma altura sonora específica (como **Dó**, **Ré**, **Mi**, **Fá**, **Sol**, **Lá**, **Si** — na cifra internacional: `C, D, E, F, G, A, B`).
- **Analogia:** é como o nome de uma cor ("azul", "vermelho"). Um `C4` (Dó central) é sempre a mesma altura, seja tocado num piano, violão ou guitarra.

### Oitava
O intervalo entre uma nota e a próxima nota com o mesmo nome que vibra exatamente com o **dobro da frequência**.
- **Analogia:** um degrau completo numa escada em espiral — você volta para o mesmo "lado" (ex.: de `C3` para `C4`), mas um andar acima. O número ao lado da letra (`C3`, `C4`, `C5`) indica qual andar (oitava) é.

### Corda
Cada um dos 6 fios metálicos tensionados na guitarra, numerados da **mais fina/aguda (corda 1 — E4)** até a **mais grossa/grave (corda 6 — E2)**.
- **Cordas lisas (*plain*):** fios sólidos de aço (normalmente cordas 1, 2 e 3).
- **Cordas encapadas (*wound*):** um núcleo fino de aço enrolado por outro fio em espiral (normalmente cordas 4, 5 e 6).

### Casa (*fret*)
Cada espaço entre os filetes metálicos (trastes) ao longo do braço da guitarra.
- **Casa 0** significa tocar a corda **solta** (sem apertar nenhum traste).
- Cada casa acima encurta o comprimento vibrante da corda e sobe o som em **1 semitom** (como avançar uma tecla no piano). Na **casa 12**, o comprimento cai exatamente pela metade e o som sobe uma oitava inteira.

### Afinação (*tuning*)
Qual nota cada corda produz quando tocada solta (casa 0). Na afinação padrão da guitarra de 6 cordas: `1: E4, 2: B3, 3: G3, 4: D3, 5: A2, 6: E2`.
- **Afinações alternativas:** como **Drop D** (abaixar a corda 6 de `E2` para `D2`) ou **Meio tom abaixo (Eb)** (todas as cordas um semitom mais graves).

### Cents
Uma unidade minúscula usada para medir desvios de afinação. **1 semitom = 100 cents** (e 1 oitava = 1200 cents).
- **Analogia:** se um semitom fosse 1 metro, 1 cent seria 1 centímetro. O ouvido humano começa a notar desafinação por volta de 5 a 10 cents; o NotesId mede $f_0$ com precisão melhor que 1 cent.

---

## Física do som e da guitarra

### Frequência (Hz)
Quantas vezes por segundo a corda (e o ar ao redor) vai e volta. Medida em **Hertz (Hz)**.
- **Exemplo:** o Lá de referência (`A4`) vibra **440 vezes por segundo (440 Hz)**. Quanto maior o número em Hz, mais agudo o som.

### Frequência fundamental ($f_0$)
A vibração principal — mais lenta e geralmente mais forte — que define **qual nota** nossos ouvidos percebem.
- **Exemplo:** quando você toca o Dó4 em qualquer uma das 5 posições da guitarra, $f_0$ é sempre ~**261,63 Hz**. É por isso que $f_0$ sozinha diz *qual nota* foi tocada, mas **não** diz *em qual corda e casa*.

### Harmônicos e Parciais
Quando uma corda vibra, ela não vibra apenas como um todo ($f_0$): ela vibra simultaneamente em metades ($2 f_0$), terços ($3 f_0$), quartos ($4 f_0$) e assim por diante.
- **Harmônicos ideais:** múltiplos inteiros exatos ($1 \cdot f_0, 2 \cdot f_0, 3 \cdot f_0, \dots$), que só existiriam numa corda perfeitamente flexível (como um fio de seda sem rigidez).
- **Parciais ($f_n$):** as frequências **reais** medidas na guitarra. O parcial de ordem $n$ corresponde ao $n$-ésimo modo de vibração.

### Inharmonicidade ($B$)
O pequeno **desvio para cima** (agudo) que os parciais reais sofrem em relação aos harmônicos ideais por causa da **rigidez física do aço** da corda:
$$f_n = n \cdot f_0 \cdot \sqrt{1 + B \cdot n^2}$$
- **Por que uma corda mais rígida ou mais curta desafina os agudos?** Imagine tentar dobrar um barbante versus uma barra fina de ferro: a barra "resiste" a fazer curvas apertadas e volta mais rápido. Como os parciais agudos ($n$ grande) exigem curvas cada vez menores ao longo da corda, a rigidez acelera essas vibrações, deixando os parciais agudos ligeiramente mais altos que $n \cdot f_0$.
- **Por que isso resolve o problema do NotesId?** Tocar o mesmo Dó4 na corda 6 casa 20 usa uma corda grossa e curtíssima ($B$ muito alto); tocar na corda 2 casa 1 usa uma corda fina e comprida ($B$ ~45× menor). O valor de $B$ funciona como a **"impressão digital física"** da posição.

### Decaimento (*decay*)
A velocidade com que o volume de cada parcial vai diminuindo depois que você palheta a corda, medida em **dB por segundo (dB/s)**.
- Cordas mais curtas/grossas e parciais mais agudos tendem a perder energia mais rápido por atrito interno do metal e absorção no traste/dedo.

### Posição de palhetada (*pluck position*)
Em qual ponto ao longo da corda a palheta bate (por exemplo, a 20% da ponte).
- **Efeito físico (filtro pente):** se você palhetar exatamente no meio da corda (50%), todos os parciais pares ($n = 2, 4, 6, \dots$) desaparecem; se palhetar a 25%, desaparecem os múltiplos de 4 ($n = 4, 8, \dots$).

---

## Processamento de sinais e estatística

### FFT (*Fast Fourier Transform* — Transformada Rápida de Fourier)
O algoritmo matemático que transforma uma gravação de som no tempo (uma linha subindo e descendo) em um **espectro de frequências** (um gráfico mostrando quanta energia existe em cada Hz).
- **Analogia:** é como um prisma de vidro que separa a luz branca nas cores do arco-íris — a FFT separa o som da guitarra nas frequências dos seus parciais.

### Janela (*window*) e *Zero-padding*
- **Janela (ex.: Blackman-Harris, Hann):** suaviza as bordas do trecho de áudio antes da FFT para evitar "vazamentos" artificiais no espectro.
- **Zero-padding:** acrescentar zeros ao final do trecho antes da FFT para gerar mais pontos intermediários no gráfico, permitindo localizar picos de frequência com altíssima precisão (junto da interpolação parabólica).

### STFT (*Short-Time Fourier Transform*)
Em vez de fazer uma única FFT no áudio inteiro, faz várias FFTs curtas em sequência ao longo do tempo.
- **Analogia:** se a FFT é uma *fotografia* das frequências, a STFT é um *filme* mostrando como cada parcial vai enfraquecendo segundo a segundo (usada para medir o decaimento).

### Distância de Mahalanobis / Classificador Gaussiano
Uma fórmula estatística clássica que mede o quão "perto" o $B$ (e o decaimento) medidos estão do valor esperado para cada candidato `(corda, casa)`, levando em conta a dispersão natural ($\sigma$) de cada medição. Quanto menor a distância, maior a pontuação do candidato.

### Calibração
O processo de tocar algumas notas conhecidas (por exemplo, cada corda solta e na casa 12) para que o sistema meça a rigidez própria ($B_0$) da **sua** guitarra e do **seu** jogo de cordas. A partir dessas poucas notas, a física ($B(\text{casa}) = B_0 \cdot 2^{\text{casa}/6}$) prevê o $B$ de todas as 22 casas.

### Confiança e `indeterminado`
Um valor de **0% a 100%** que indica o quanto o candidato vencedor se destacou dos concorrentes. Quando dois candidatos ficam muito empatados (por exemplo, `corda 3 casa 5` vs `corda 4 casa 10`), o NotesId prefere responder **`indeterminado`** com honestidade em vez de chutar.

### DI (linha direta) vs Microfone
- **DI / Interface de áudio (entrada Hi-Z / Instrumento):** captura o sinal elétrico puro do captador da guitarra pelo cabo P10, sem eco da sala nem ruído ambiente — é o cenário ideal para o NotesId.
- **Microfone (notebook/tablet/celular):** captura o som pelo ar, misturando ruído ambiente, reverberação da sala e filtros de voz do sistema operacional, o que reduz a precisão na medição de $B$.

---

## Rede local e HTTPS

### IP local / Rede local (LAN)
O endereço privado que o roteador Wi-Fi atribui ao seu computador dentro da rede da sua casa ou estúdio (por exemplo, `192.168.0.10`).
- **Detalhe importante:** esse endereço só existe **dentro do seu Wi-Fi** — ele não é um endereço público na internet. Outros aparelhos na mesma rede (como o seu celular) conseguem falar direto com o computador sem enviar o áudio para fora.

### HTTPS e contexto seguro
`HTTPS` é a versão criptografada do `HTTP` (assim como `WSS` é a versão criptografada do `WS` para WebSockets).
- **Por que o NotesId precisa disso no celular:** navegadores modernos tratam `http://localhost` como um **contexto seguro**, então o microfone funciona no mesmo computador em HTTP simples. Quando você acessa pelo IP local (como `192.168.0.10`) a partir de um celular, o navegador **bloqueia o acesso ao microfone (`getUserMedia` e `AudioWorklet`)** a menos que a página use `HTTPS`.

### Certificado autoassinado
Um certificado digital TLS gerado localmente pelo seu próprio computador, em vez de ser emitido por uma autoridade certificadora pública (como Let's Encrypt ou DigiCert).
- **Por que o navegador mostra um aviso:** o navegador verifica se alguma autoridade conhecida atesta a identidade do servidor. Como foi o seu próprio PC quem gerou o certificado para o seu IP local, aparece um alerta como *"Sua conexão não é particular"*. Esse aviso é esperado **apenas** ao acessar o seu próprio servidor local do NotesId — **nunca** aceite esse aviso em sites desconhecidos.

### Firewall
Uma barreira de segurança embutida no Windows (Firewall do Windows Defender) e no Ubuntu (`ufw`) que bloqueia, por padrão, conexões recebidas da rede.
- Para que o celular consiga alcançar o NotesId na porta `8000`, você libera temporariamente a entrada na porta `8000` apenas na sua **rede Privada / local**, e pode remover a regra assim que terminar o teste.
