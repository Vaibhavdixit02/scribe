from setuptools import setup, find_packages

setup(
    name="scribe_agent",
    version="0.1.0",
    description="Cross-Modal Web Agent for improved web navigation",
    author="Scribe Team",
    author_email="team@scribecorp.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "peft>=0.4.0",
        "vllm>=0.1.4",
        "datasets>=2.12.0",
        "beautifulsoup4>=4.11.1",
        "pillow>=9.4.0",
        "opencv-python>=4.7.0",
        "wandb>=0.15.0",
        "tensorboard>=2.12.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "accelerate>=0.19.0",
        "einops>=0.6.0",
    ],
    python_requires=">=3.8",
)