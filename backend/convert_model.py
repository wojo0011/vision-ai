import sys
import pathlib
import subprocess
import importlib.util


_MISSING_TFJS_MSG = (
    "TensorFlow.js converter is not installed. "
    "Install it with: pip install tensorflowjs\n"
)


def convert_keras_to_tfjs(keras_path, output_dir):
    """
    Convert a Keras model to TensorFlow.js format.
    """
    keras_path = str(pathlib.Path(keras_path).resolve())
    output_dir = str(pathlib.Path(output_dir).resolve())

    if importlib.util.find_spec("tensorflowjs") is None:
        return (1, "", _MISSING_TFJS_MSG)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        sys.executable,
        '-m',
        'backend.tfjs_compat_converter',
        '--input_format=keras', keras_path, output_dir
    ], capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_model.py <keras_model_path> <output_dir>")
        sys.exit(1)
    code, out, err = convert_keras_to_tfjs(sys.argv[1], sys.argv[2])
    print(out)
    if err:
        print(err, file=sys.stderr)
    sys.exit(code)
