from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
	scripts=["node/hri_face_detect/detect"],
	#packages=['hri_face_detect'],
	#package_dir={'': 'src'}
	)

setup(**d)

