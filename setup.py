import os
import sys
import shutil
import glob
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel

PROTOBUF_LIB_DIR = "/opt/protobuf"

# Create the .pth file in the source root so setuptools can package it as data_files
with open('diffusion_graph_paths.pth', 'w') as f:
    f.write('external/torch-mlir/python_packages/torch_mlir\n')

def _auditwheel_repair(wheel_path, wheel_dir):
    """
    Run `auditwheel repair` on *wheel_path* and write the repaired wheel into
    *wheel_dir*.  Falls back gracefully if auditwheel is not installed or if
    we are not on Linux (macOS users should use delocate instead — see note).
    """
    if sys.platform != "linux":
        print(
            "auditwheel is Linux-only.  On macOS run:\n"
            "  pip install delocate && delocate-wheel -w dist/ <wheel>\n"
            "Skipping RPATH repair — the wheel may not be portable."
        )
        return wheel_path
 
    if shutil.which("auditwheel") is None:
        print(
            "auditwheel not found — installing it now via pip ...\n"
            "(add 'auditwheel' to your build dependencies to avoid this)"
        )
        subprocess.check_call([sys.executable, "-m", "pip", "install", "auditwheel"])
 
    print(f"\nauditwheel repair  →  {wheel_dir}")
    subprocess.check_call(
        ["auditwheel", "repair", wheel_path, "--wheel-dir", wheel_dir],
        # Make sure the protobuf libs are visible to auditwheel's ldd scan
        env={**os.environ, "LD_LIBRARY_PATH": f"{PROTOBUF_LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"},
    )
 
    # Return the path of the repaired wheel (auditwheel writes one file)
    repaired = glob.glob(os.path.join(wheel_dir, "*.whl"))
    if repaired:
        print(f"Repaired wheel: {repaired[0]}")
        return repaired[0]
 
    print("Warning: auditwheel produced no output wheel — check its log above.")
    return wheel_path


class CustomBuildPy(build_py):
    def run(self):
        # 1. Compile the C++ compiler and torch-mlir
        print("Starting custom compilation step...")
        workspace_dir = os.path.abspath(os.path.dirname(__file__))
        home_diffusion_project = os.path.expanduser("~/diffusion-project")
        created_symlink = False
        
        # Ensure the ~/diffusion-project symlink exists so scripts/build succeeds
        if not os.path.exists(home_diffusion_project):
            try:
                os.symlink(workspace_dir, home_diffusion_project)
                created_symlink = True
                print(f"Created symlink ~/diffusion-project -> {workspace_dir}")
            except Exception as e:
                print(f"Warning: could not create symlink ~/diffusion-project: {e}")
                
        try:
            print("Executing build_compiler command...")
            subprocess.check_call(['bash', '-c', f'source {home_diffusion_project}/scripts/build && build_compiler'])
        finally:
            if created_symlink:
                try:
                    os.unlink(home_diffusion_project)
                    print("Removed symlink ~/diffusion-project")
                except Exception:
                    pass

        # 2. Run the standard build process
        super().run()

        # 3. Copy compiler.py and the compiled graph_compiler.so to self.build_lib
        print("Copying compiler files to build_lib...")
        shutil.copy(f'{home_diffusion_project}/compiler/python/compiler.py', os.path.join(self.build_lib, 'compiler.py'))
        
        so_files = glob.glob(f'{home_diffusion_project}/build/python/graph_compiler*.so')
        if not so_files:
            raise RuntimeError("Could not find compiled graph_compiler extension in build/python")
        for so_file in so_files:
            dest = os.path.join(self.build_lib, os.path.basename(so_file))
            shutil.copy(so_file, dest)
            print(f"Copied {so_file} to {dest}")

        # 4. Copy build/external/torch-mlir to self.build_lib/external/torch-mlir
        print("Copying torch-mlir build files to self.build_lib/external/torch-mlir...")
        src_external = f'{home_diffusion_project}/build/external/torch-mlir/python_packages/torch_mlir/torch_mlir'
        dest_external = os.path.join(self.build_lib, 'external', 'torch_mlir')
        
        if os.path.exists(dest_external):
            shutil.rmtree(dest_external)
            
        # copytree with symlinks=False will resolve symlinks to actual files
        shutil.copytree(
            src_external, 
            dest_external, 
            symlinks=False, 
            ignore_dangling_symlinks=True,
            dirs_exist_ok=True
        )
        print("torch-mlir files copied successfully.")

        # 5. Copy patterns directory to self.build_lib/Patterns
        print("Copying patterns directory to self.build_lib/Patterns...")
        src_patterns = f'{home_diffusion_project}/build/Patterns'
        dest_patterns = os.path.join(self.build_lib, 'Patterns')
        
        if os.path.exists(dest_patterns):
            shutil.rmtree(dest_patterns)
            
        shutil.copytree(
            src_patterns, 
            dest_patterns, 
            symlinks=False, 
            ignore_dangling_symlinks=True,
            dirs_exist_ok=True
        )
        print("Patterns copied successfully.")

        # 6. Copying shared protobuf libs to self.build_lib
        print("Copying shared protobuf libs to self.build_lib...")
        so_files = glob.glob('/opt/protobuf/libprotobuf*')

        for so_file in so_files:
            dest = os.path.join(self.build_lib, os.path.basename(so_file))
            shutil.copy(so_file, dest)
            print(f"Copied {so_file} to {dest}")
        
        print("Protobuf libs copied successfully.")

        # 7. Copying compiled protobuf python msg

        src_file = f"{home_diffusion_project}/build/python/GraphConvertor/weightBuffers_pb2.py"
        dest_dir = os.path.join(self.build_lib, "GraphConvertor")
        os.makedirs(dest_dir, exist_ok=True)
        dest_file = os.path.join(dest_dir, "weightBuffers_pb2.py")
        shutil.copy(src_file, dest_file)
        print(f"Copied {src_file} to {dest_file}")

class AuditwheelBdistWheel(bdist_wheel):
    """
    Extends the standard wheel build to run `auditwheel repair` automatically.
 
    Flow:
      python setup.py bdist_wheel            ← triggers this class
        → calls super().run()               ← produces a raw linux_x86_64 wheel
        → calls _auditwheel_repair()        ← rewrites RPATH, bundles .so files
        → prints path of manylinux wheel
 
    The original (unrepaired) wheel is deleted so only the portable wheel
    ends up in dist/.
    """
 
    def run(self):
        super().run()   # produces dist/*.whl as usual
 
        raw_wheels = glob.glob(os.path.join(self.dist_dir, "*.whl"))
        if not raw_wheels:
            print("No wheel found in dist/ — skipping auditwheel step.")
            return
 
        for raw_wheel in raw_wheels:
            # Skip wheels that are already manylinux-tagged (e.g. on re-run)
            if "manylinux" in os.path.basename(raw_wheel):
                print(f"Already manylinux-tagged, skipping: {raw_wheel}")
                continue
 
            repaired = _auditwheel_repair(raw_wheel, self.dist_dir)
 
            # Remove the raw wheel so only the repaired one lives in dist/
            if repaired != raw_wheel and os.path.exists(raw_wheel):
                os.remove(raw_wheel)
                print(f"Removed unrepaired wheel: {raw_wheel}")


setup(
    packages=find_packages(where='diffusion_graph'),
    package_dir={'': 'diffusion_graph'},
    cmdclass={
        'build_py': CustomBuildPy,
        'bdist_wheel': AuditwheelBdistWheel,
    },
    data_files=[
        ('.', ['diffusion_graph_paths.pth'])
    ],
    zip_safe=False
)
