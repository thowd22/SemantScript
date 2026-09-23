from __future__ import annotations

import semantscript_model
import semantscript_model.calibration as calibration_module
import semantscript_model.classifier as classifier_module
import semantscript_model.encoder as encoder_module
import semantscript_model.heads as heads_module
import semantscript_model.losses as losses_module


def test_top_level_package_exports_model_public_api() -> None:
    modules = (
        calibration_module,
        classifier_module,
        encoder_module,
        heads_module,
        losses_module,
    )

    for module in modules:
        for name in module.__all__:
            assert name in semantscript_model.__all__
            assert getattr(semantscript_model, name) is getattr(module, name)
