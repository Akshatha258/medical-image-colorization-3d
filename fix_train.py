"""
Quick fix script to update train.py
Run this to fix the use_attention parameter issue
"""

import re

# Read the train.py file
with open('training/train.py', 'r') as f:
    content = f.read()

# Replace the get_model call
old_pattern = r"""model = get_model\(
        model_type=config\.MODEL_TYPE,
        use_attention=config\.USE_ATTENTION,
        base_features_3d=config\.BASE_FEATURES_3D,
        num_slices=config\.NUM_SLICES
    \)"""

new_code = """model = get_model(
        model_type=config.MODEL_TYPE,
        base_features_3d=config.BASE_FEATURES_3D,
        num_slices=config.NUM_SLICES
    )"""

# Replace
content = re.sub(old_pattern, new_code, content)

# Write back
with open('training/train.py', 'w') as f:
    f.write(content)

print("✓ Fixed train.py - removed use_attention parameter")
