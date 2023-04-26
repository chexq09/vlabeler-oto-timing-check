# Oto Timing Check - vLabeler plugin

## Requirement

This is a batch edit plugin for vLabeler. It does similar things as setParam's Utterance Timing (F8), which checks the utterance setting by adding metronome clicks to the entry audio.

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

## Download

Please check the latest release and download the plugin package which is prepared for your OS.

## Installation

See [here](https://github.com/sdercolin/vlabeler#batch-edit-plugins).
Please unzip the package to the plugin directory and import it.

## Metronome

Metronome voice comes from [here](https://pixabay.com/sound-effects/metronome-05-67359/) ([license](https://pixabay.com/service/license-summary/)). 
The sound is cut to only one click, and its volume becomes larger.

## TODO

- [x] Resample metronome.
- [x] Plugin.
- [x] Description.
- [x] Release.
