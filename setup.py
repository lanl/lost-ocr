"""Setup configuration for Lost in OCR Translation package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lost-ocr",
    version="1.0.0",
    author="Alex Most, Manish Bhattarai",
    author_email="amost@lanl.gov",
    description="Vision-based approaches to robust document retrieval under OCR degradation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lanl/lost-ocr",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ocr-extract=scripts.samba_llama:main",
            "generate-qa=scripts.new_qa_datagen:main",
            "evaluate-rag=scripts.rag_llama_measurement:main",
            "evaluate-nougat=scripts.nougat_rag:main",
            "evaluate-vision=scripts.vidore_vlm:main",
            "compare-ocr=scripts.levenshtein_per_page_by_degradation:main",
        ],
    },
    project_urls={
        "Paper": "https://arxiv.org/abs/2505.05666",
        "Bug Reports": "https://github.com/yourusername/lost-ocr/issues",
        "Source": "https://github.com/yourusername/lost-ocr",
    },
)
