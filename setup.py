import os
import shutil
import glob
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

# Create the .pth file in the source root so setuptools can package it as data_files
with open('diffusion_graph_paths.pth', 'w') as f:
    f.write('external/torch-mlir/python_packages/torch_mlir\n')

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

setup(
    packages=find_packages(where='diffusion_graph'),
    package_dir={'': 'diffusion_graph'},
    cmdclass={
        'build_py': CustomBuildPy,
    },
    data_files=[
        ('.', ['diffusion_graph_paths.pth'])
    ],
    zip_safe=False
)
