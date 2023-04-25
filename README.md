# Oto Timing Check - vLabeler plugin

## Requirement

### Compile

- Python >= 3.7 (to use PyInstaller)
- PyInstaller

Other libraries are not used to reduce the size of the package.

### As plugin

vLabeler 1.0.0-beta4 or later.

## Compile

Compile by

```
cd src
pyinstaller -F --clean ./timing_check.py
```

It is recommended to use a new venv for smaller executables.

## Metronome

Metronome voice comes from [here](https://pixabay.com/sound-effects/metronome-05-67359/) ([license](https://pixabay.com/service/license-summary/)). 
The sound is cut to only one click, and its volume becomes larger.

## TODO

- [x] Resample metronome.
- [x] Plugin.
- [ ] Description.
