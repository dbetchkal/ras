# Representative Acoustic States
Representative Acoustic States (RAS): a general, conceptually principled method to partition an arbitrary spectrogram into a structured vocabulary of states.

## Installation

The repository has been tested with Python 3.12. Runtime dependencies are declared in `pyproject.toml` and installed automatically by pip.

Clone this repository and also [`stsw`](https://github.com/lilythchu/STSW). Optionally, clone [`nps_active_space`](https://github.com/dbetchkal/NPS-ActiveSpace).

```bash
git clone https://github.com/dbetchkal/ras.git
git clone https://github.com/lilythchu/STSW.git
git clone https://github.com/dbetchkal/NPS-ActiveSpace.git
cd ras
```

Create virtual environment with updated pip
```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

Install required Python packages
```bat
python -m pip install -e ".[dev,stsw,plot]"
```

## Quick start

```
model = RAS(RASConfig())    # default configuration
result = model.analyze(S)   # `S` is a DataFrame with DatetimeIndex and band center-frequency columns
```
A `RAS` analysis `result` centrally includes the following objects:
```
S_E = result.S_E            # Ambient-relative spectral contrast
M = result.M                # Hypersphere of spectral directions
X_N = result.X_N            # Representative Acoustic States matrix
G = result.G_MST            # Minimum Spanning Tree graph
S_Pi = result.S_Pi          # Spectrogram, RAS-partitioned
```
These trace the process: 
```math
S \to S_E \to M \to X_N \to G \to \Pi
```

## License

### Public domain

This project is in the worldwide [public domain](LICENSE.md):

> This project is in the public domain within the United States,
> and copyright and related rights in the work worldwide are waived through the
> [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
>
> All contributions to this project will be released under the CC0 dedication.
> By submitting a pull request, you are agreeing to comply with this waiver of copyright interest.
