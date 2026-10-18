import { app } from "../../../scripts/app.js";

// AMDNR v0.3.5.6 Official Style Presets
const PRESETS = {
    "Full Model Look": {
        "residual_strength": 1.00,
        "residual_limit": 0.00,
        "temporal_stability": 0.50,
        "sharpening": 0.00,
        "detail_strength": 2.00,
        "colour_strength": 1.00,
        "neural_placement": "After upscaling",
        "full_network": true
    },
    "NVIDIA look": {
        "residual_strength": 1.00,
        "residual_limit": 0.00,
        "temporal_stability": 0.60,
        "sharpening": 0.00,
        "detail_strength": 1.00,
        "colour_strength": 1.00,
        "neural_placement": "Before upscaling",
        "full_network": true
    },
    "AMDNR soft": {
        "residual_strength": 0.70,
        "residual_limit": 0.15,
        "temporal_stability": 0.70,
        "sharpening": 0.00,
        "detail_strength": 1.00,
        "colour_strength": 1.00,
        "neural_placement": "Before upscaling",
        "full_network": false
    },
    "Cinematic": {
        "residual_strength": 0.85,
        "residual_limit": 0.20,
        "temporal_stability": 0.75,
        "sharpening": 0.00,
        "detail_strength": 1.00,
        "colour_strength": 1.00
    },
    "Crisp": {
        "residual_strength": 1.20,
        "residual_limit": 0.35,
        "temporal_stability": 0.50,
        "sharpening": 0.30,
        "detail_strength": 1.00,
        "colour_strength": 1.00
    },
    "Natural": {
        "residual_strength": 0.70,
        "residual_limit": 0.15,
        "temporal_stability": 0.70,
        "sharpening": 0.00,
        "detail_strength": 1.00,
        "colour_strength": 1.00
    },
    "Vivid": {
        "residual_strength": 1.10,
        "residual_limit": 0.30,
        "temporal_stability": 0.60,
        "sharpening": 0.15,
        "detail_strength": 1.00,
        "colour_strength": 1.00
    },
    "Default": {
        "residual_strength": 1.00,
        "residual_limit": 0.00,
        "temporal_stability": 0.50,
        "sharpening": 0.00,
        "detail_strength": 2.00,
        "colour_strength": 1.00,
        "neural_placement": "After upscaling",
        "full_network": true
    }
};

app.registerExtension({
    name: "ComfyUI.RH_DLSS5_AMD",
    async nodeCreated(node) {
        if (!node.comfyClass || (!node.comfyClass.includes("RH_DLSS5") && !node.comfyClass.includes("Dlss5Amd"))) {
            return;
        }

        // Style selector widget
        const styleWidget = node.widgets ? node.widgets.find(w => w.name === "nr_style" || w.name === "NR style") : null;
        if (!styleWidget) return;

        let isApplyingPreset = false;

        const origCallback = styleWidget.callback;
        styleWidget.callback = function(value) {
            if (PRESETS[value]) {
                isApplyingPreset = true;
                const p = PRESETS[value];
                for (const key in p) {
                    const w = node.widgets.find(w2 =>
                        w2.name === key ||
                        w2.name === key.replace(/_/g, " ") ||
                        w2.name.toLowerCase() === key.toLowerCase()
                    );
                    if (w) {
                        w.value = p[key];
                    }
                }
                isApplyingPreset = false;
                if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
            }
            if (origCallback) {
                return origCallback.apply(this, arguments);
            }
        };

        // If user manually tweaks any slider, automatically switch style to "Custom"
        const sliderKeys = [
            "residual_strength", "residual_limit", "temporal_stability",
            "sharpening", "detail_strength", "colour_strength"
        ];
        sliderKeys.forEach(key => {
            const w = node.widgets.find(w2 =>
                w2.name === key ||
                w2.name === key.replace(/_/g, " ") ||
                w2.name.toLowerCase() === key.toLowerCase()
            );
            if (w) {
                const origSliderCb = w.callback;
                w.callback = function() {
                    if (!isApplyingPreset && styleWidget.value !== "Custom") {
                        styleWidget.value = "Custom";
                        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
                    }
                    if (origSliderCb) {
                        return origSliderCb.apply(this, arguments);
                    }
                };
            }
        });
    }
});
