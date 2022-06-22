from setuptools import find_packages, setup


install_requires = [
    'numpy==1.22.0',
    'pandas==1.0.3',
    'scikit-learn==0.22.2.post1',
    'pytest==5.4.1',
    'pytest-cov==2.8.1',
    'joblib==0.14.1',
]


setup(
    name='auto_ml',
    version='0.0.1',
    description='Auto ml lib to run different ml models',
    platforms=['POSIX'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    zip_safe=False
)