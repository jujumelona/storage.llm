# Model Profiles

The runtime pipeline is shared, but model tensor naming and cache units are not.
Each model gets a profile directory with:

- tensor naming rules
- cache unit definition
- default transfer pipeline
- required sidecars
- topology learning path

Keep format- or model-specific codec policy in separate profile directories
instead of mixing naming rules into the core engine. The runtime must prefer
metadata from the loaded model root over profile names whenever the format
provides the needed fields.
