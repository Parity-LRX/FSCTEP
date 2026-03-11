"""Setup script for molecular_force_field package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if exists
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

setup(
    name="molecular_force_field",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python library for molecular modeling with E3NN-based neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/molecular_force_field",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        # Keep compatible with mace-torch (often pins e3nn==0.4.4)
        "e3nn>=0.4.4,<0.6.0",
        "ase>=3.22.0",
        "matscipy>=0.8.0",
        "tqdm>=4.62.0",
        "h5py>=3.6.0",
        "tables>=3.8.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        # Optional PyG extensions (faster scatter / radius graph). The codebase has
        # pure-PyTorch fallbacks, but these are recommended when wheels are available.
        "pyg": [
            "torch-scatter>=2.0.9",
            "torch-cluster>=1.6.0",
        ],
        # Optional backend: cuEquivariance (for tensor_product_mode="spherical-save-cue").
        # This is kept optional to avoid pip upgrading torch in environments that pin torch/torchvision/torchaudio.
        "cue": [
            "cuequivariance-torch>=0.8.1",
            "cuequivariance-ops-torch-cu12>=0.8.1; platform_system=='Linux'",
        ],
        # Convenience extra.
        "full": [
            "torch-scatter>=2.0.9",
            "torch-cluster>=1.6.0",
            "cuequivariance-torch>=0.8.1",
            "cuequivariance-ops-torch-cu12>=0.8.1; platform_system=='Linux'",
            # Active learning diversity (SOAP, FPS helpers)
            "dscribe>=2.0.0",
            "scikit-learn>=1.0.0",
        ],
        # Active learning (PES coverage, etc.)
        "al": [
            "dscribe>=2.0.0",
            "scikit-learn>=1.0.0",
        ],
        # Alias for users who only want SOAP deps
        "soap": [
            "dscribe>=2.0.0",
        ],
        # Thermal transport (IFC2/IFC3 -> phono3py BTE -> Callaway engineering scattering)
        "thermal": [
            "phono3py>=2.0.0",
            "scipy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mff-train=molecular_force_field.cli.train:main",
            "mff-evaluate=molecular_force_field.cli.evaluate:main",
            "mff-preprocess=molecular_force_field.cli.preprocess:main",
            "mff-lammps=molecular_force_field.cli.lammps_interface:main",
            "mff-export-core=molecular_force_field.cli.export_libtorch_core:main",
            "mff-evaluate-pes-coverage=molecular_force_field.cli.evaluate_pes_coverage:main",
            "mff-active-learn=molecular_force_field.cli.active_learning:main",
            "mff-init-data=molecular_force_field.cli.init_data:main",
        ],
    },
)