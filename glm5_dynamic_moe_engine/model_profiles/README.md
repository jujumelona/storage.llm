# Model Profiles

The runtime pipeline is shared, but model tensor naming and cache units are not.
Each model gets a profile directory with:

- tensor naming rules
- cache unit definition
- default transfer pipeline
- required sidecars
- topology learning path

Keep GLM-5.1-specific codec policy in `glm5_1_nvfp4`. Add other models as
separate profile directories instead of mixing naming rules into the core
engine.
