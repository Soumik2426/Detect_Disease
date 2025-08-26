import os
import tensorflow as tf
import shutil

def convert_model(model_path, model_type, version):
    """Convert a model to TensorFlow Serving format with specified version."""
    try:
        print(f"🔄 Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully: {model_path}")

        # Ensure input shape is defined
        if model.input_shape is None:
            raise ValueError("Model does not have a defined input shape.")

        input_shape = model.input_shape[1:]  # Remove batch dimension

        # Create serving function
        @tf.function(input_signature=[tf.TensorSpec(shape=[None] + list(input_shape), dtype=tf.float32)])
        def serving_fn(input_tensor):
            return model(input_tensor)

        # Define export directory
        output_dir = os.path.join(BETA_DIR if model_type == 'beta' else PRODUCTION_DIR, str(version))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save model in TensorFlow Serving format
        tf.saved_model.save(model, output_dir, signatures={"serving_default": serving_fn})

        print(f"✅ Successfully converted {os.path.basename(model_path)} to {model_type} (v{version})")
        return True

    except Exception as e:
        print(f"❌ Failed to convert {os.path.basename(model_path)}: {str(e)}")
        return False
