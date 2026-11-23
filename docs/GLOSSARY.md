# Glossary — NotesId

> 🇺🇸 **English** | 🇧🇷 [Versão em Português](GLOSSARY.pt-BR.md)

This glossary explains the musical, physical, and signal-processing terms used in **NotesId** in plain language, assuming no prior background in music theory or audio engineering.

---

## Basic Musical Terms

### Note
The name given to a specific musical pitch (such as `C, D, E, F, G, A, B`).
- **Analogy:** like the name of a color ("blue", "red"). Middle C (`C4`) is always the same pitch whether played on a piano or a guitar.

### Octave
The interval between one note and the next note with the same name vibrating at **exactly twice the frequency**.
- **Analogy:** going one full loop up a spiral staircase — you face the same direction (`C3` → `C4`), one floor higher. The number next to the letter (`C3`, `C4`, `C5`) tells you which floor (octave).

### String
One of the 6 tensioned metal wires on a guitar, numbered from **thinnest/highest (string 1 — E4)** to **thickest/lowest (string 6 — E2)**.
- **Plain strings:** solid steel wire (typically strings 1, 2, and 3).
- **Wound strings:** a thin steel core wrapped with a spiral outer wire (typically strings 4, 5, and 6).

### Fret
Each space between the metal bars along the guitar neck.
- **Fret 0** means playing the **open** string (no finger pressing down).
- Each fret shortens the vibrating string and raises the pitch by **1 semitone** (one adjacent key on a piano). At **fret 12**, the string length is halved and the pitch rises by a full octave.

### Tuning
The pitch each string produces when played open (fret 0). Standard 6-string guitar tuning is `1: E4, 2: B3, 3: G3, 4: D3, 5: A2, 6: E2`.
- **Alternate tunings:** e.g., **Drop D** (lowering string 6 from `E2` to `D2`) or **Half-step down (Eb)** (every string lowered by one semitone).

### Cents
A fine unit for measuring pitch differences. **1 semitone = 100 cents** (1 octave = 1200 cents).
- **Analogy:** if one semitone were 1 meter, 1 cent would be 1 centimeter. Most people start noticing detuning around 5–10 cents; NotesId measures $f_0$ to sub-cent precision.

---

## Sound & Guitar Physics

### Frequency (Hz)
How many times per second the string (and air) vibrates back and forth, measured in **Hertz (Hz)**.
- **Example:** standard concert A (`A4`) vibrates **440 times per second (440 Hz)**. Higher Hz = higher pitch.

### Fundamental Frequency ($f_0$)
The slowest, primary vibration that determines **which note** we perceive.
- **Example:** playing `C4` at any of its 5 guitar positions produces $f_0 \approx 261.63\text{ Hz}$. That is why $f_0$ alone tells you *which note* was played, but **not** *which string and fret*.

### Harmonics & Partials
A vibrating string oscillates not only as a whole ($f_0$), but simultaneously in halves ($2 f_0$), thirds ($3 f_0$), quarters ($4 f_0$), etc.
- **Ideal harmonics:** exact integer multiples ($1 f_0, 2 f_0, 3 f_0, \dots$), which only exist in a hypothetical perfectly flexible string.
- **Partials ($f_n$):** the **actual** overtone frequencies measured on a real guitar. Order $n$ is the $n$-th vibration mode.

### Inharmonicity ($B$)
The slight **upward pitch shift** of real partials relative to ideal harmonics caused by the **bending stiffness of steel**:
$$f_n = n \cdot f_0 \cdot \sqrt{1 + B \cdot n^2}$$
- **Why does a stiffer or shorter string sharpen upper partials?** Imagine bending a soft shoelace vs. a thin metal rod: the rod resists tight bends and snaps back faster. Because higher partials ($n$ large) have tighter bends along the string, stiffness accelerates them, pushing their frequencies slightly above $n \cdot f_0$.
- **Why does this solve NotesId's problem?** Playing `C4` on string 6 fret 20 uses a thick, very short string (high $B$); playing it on string 2 fret 1 uses a thin, long string (~45× smaller $B$). Thus $B$ acts as a **physical fingerprint** of the string-fret position.

### Decay
How fast each partial loses loudness after being plucked, measured in **dB per second (dB/s)**.
- Shorter/thicker strings and higher partials lose energy faster due to internal metal damping and fret/finger losses.

### Pluck Position
Where along the string the pick strikes (e.g., 20% from the bridge).
- **Physical effect (comb filter):** plucking at the exact midpoint (50%) suppresses all even partials ($n = 2, 4, 6, \dots$); plucking at 25% suppresses multiples of 4 ($n = 4, 8, \dots$).

---

## Signal Processing & Classical Statistics

### FFT (Fast Fourier Transform)
The mathematical algorithm that turns a time-domain audio wave into a **frequency spectrum** (how much energy exists at every Hz).
- **Analogy:** like a glass prism splitting white light into rainbow colors — FFT splits a guitar note into its individual partial frequencies.

### Window & Zero-padding
- **Window (e.g., Blackman-Harris, Hann):** smoothly tapers the start and end of an audio slice before FFT to prevent artificial spectral "leakage".
- **Zero-padding:** appending zeros before the FFT to create a denser frequency grid, enabling pinpoint peak localization alongside parabolic interpolation.

### STFT (Short-Time Fourier Transform)
Running many short FFTs back-to-back over time instead of one big FFT.
- **Analogy:** if FFT is a *photograph* of frequencies, STFT is a *movie* showing how each partial fades over time (used to measure decay rates).

### Mahalanobis Distance / Gaussian Classifier
A classical statistical formula measuring how close the measured $B$ (and decay) is to the expected value of each `(string, fret)` candidate, scaled by each feature's natural spread ($\sigma$). Smaller distance = higher candidate score.

### Calibration
Playing a handful of known notes (e.g., each open string and fret 12) so NotesId measures the inherent stiffness ($B_0$) of **your** guitar and strings. From those few notes, physics ($B(\text{fret}) = B_0 \cdot 2^{\text{fret}/6}$) predicts $B$ for all 22 frets.

### Confidence & `indeterminate`
A **0%–100%** score showing how clearly the top candidate beats the runners-up. When two candidates are physically near-tied (such as `string 3 fret 5` vs. `string 4 fret 10`), NotesId honestly reports **`indeterminate`** instead of guessing blindly.

### DI (Direct Input) vs. Microphone
- **DI / Audio Interface (Hi-Z / Instrument input):** captures the pure electrical signal from the guitar pickup via a 1/4" cable with no room echo or ambient noise — ideal for NotesId.
- **Microphone (laptop/tablet):** captures airborne sound mixed with room reverb, ambient noise, and OS voice filters, reducing $B$ measurement accuracy.
