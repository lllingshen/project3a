from setuptools import find_packages, setup

setup(
    name="tb3_locateanything",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/tb3_locateanything"]),
        ("share/tb3_locateanything", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ling Shen",
    maintainer_email="lllingshenmail@gmail.com",
    description="Command-triggered LocateAnything grounding for the course navigation backend.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "locateanything_node = tb3_locateanything.locateanything_node:main",
    ]},
)
