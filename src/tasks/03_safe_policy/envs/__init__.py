from isaaclab_tasks.utils import import_packages

# Mirrors the submodule convention: importing this package walks the task sub-packages so that
# their gym registrations run. Training scripts only need `import envs`.
import_packages(__name__)
